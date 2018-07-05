/* * * * *
 *  AzpLmAdam.hpp 
 *  Copyright (C) 2017 Rie Johnson
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * * * * */

#ifndef _AZP_LM_ADAM_HPP_
#define _AZP_LM_ADAM_HPP_

#include "AzpLmSgd.hpp"

/* 
 * Adam by (Kingma & Ba, ICLR 2015)
 */

class AzpLmAdam_Param {
public:
  double eta; /* alpha */
  double b1, b2, coeff, eps, grad_clip_after; 
  bool do_shrink, do_adam_var; 
  AzpLmAdam_Param() : b1(-1), b2(-1), eta(-1), coeff(1), eps(1e-8), grad_clip_after(-1), do_shrink(false), do_adam_var(false) {}

  /*------------------------------------------------------------*/ 
  #define kw_b1  "adam_b1="
  #define kw_b2  "adam_b2="  
  #define kw_adam_eps "adam_eps="
  #define kw_grad_clip_after "grad_clip_after="
  #define kw_do_shrink "Shrink"
  #define kw_do_adam_var "AdamVar"
  void resetParam(AzParam &azp, const char *pfx, bool is_warmstart=false) {
    azp.reset_prefix(pfx); 
    azp.vFloat(kw_eta, &eta); 
    azp.vFloat(kw_b1, &b1); 
    azp.vFloat(kw_b2, &b2);     
    azp.vFloat(kw_adam_eps, &eps); 
    eps = MAX(eps, azc_epsilon); 
    azp.vFloat(kw_grad_clip_after, &grad_clip_after); 
    azp.swOn(&do_shrink, kw_do_shrink); 
    azp.swOn(&do_adam_var, kw_do_adam_var); 
    azp.reset_prefix(); 
  }  

  void checkParam(const char *pfx) {  
    const char *eyec = "AzpLmAdam_Param::checkParam";   
    AzXi::throw_if_negative(b1, eyec, kw_b1, pfx); 
    AzXi::throw_if_negative(b2, eyec, kw_b2, pfx);     
    AzXi::throw_if_nonpositive(eta, eyec, kw_eta, pfx); 
    AzX::throw_if((b1 >= 1), AzInputError, eyec, kw_b1, "must be no greater than 1.");
    AzX::throw_if((b2 >= 1), AzInputError, eyec, kw_b2, "must be no greater than 1.");
  }

  void printParam(const AzOut &out, const char *pfx) const {
    if (out.isNull()) return; 
    AzPrint o(out, pfx); 
    o.printV(kw_eta, eta); 
    o.printV(kw_b1, b1); 
    o.printV(kw_b2, b2);     
    o.printV(kw_adam_eps, eps); 
    o.printV(kw_grad_clip_after, grad_clip_after); 
    o.printSw(kw_do_shrink, do_shrink); 
    o.printSw(kw_do_adam_var, do_adam_var); 
    o.printEnd(); 
  } 
}; 

class AzpLmAdam : public virtual AzpLmSgd {
protected:
  AzPmat m_w_g2avg, v_i_g2avg; 
  double tt; 

public:
  AzpLmAdam() : tt(0) {}
  virtual const char *description() const { return "Adam"; }

  virtual void resetWork() {
    AzpLmSgd::resetWork(); 
    m_w_g2avg.zeroOut();  v_i_g2avg.zeroOut(); 
    tt = 0; 
  }
  virtual void reformWork() {
    AzpLmSgd::reformWork(); 
    m_w_g2avg.reform_tmpl(&m_w); v_i_g2avg.reform_tmpl(&v_i);   
    tt = 0;   
  }  
  
  void reset(const AzpLmAdam *inp) { /* copy */
    AzpLmSgd::reset(inp); 
    m_w_g2avg.set(&inp->m_w_g2avg); v_i_g2avg.set(&inp->v_i_g2avg);   
    tt = inp->tt; 
  }                          

  void flushDelta(const AzpLmParam &p, const AzpLmAdam_Param &pa) { _flushDelta(p, pa); }  

protected:  
  void _flushDelta(const AzpLmParam &p, const AzpLmAdam_Param &pa) {  
    if (p.dont_update()) return; 
    if (grad_num <= 0) {
      return; 
    }
    check_ws("AzpLmAdam::flushDelta"); 
    
    bool do_reg = true; 
    adam_update(&m_w, &m_w_grad, grad_num, &m_w_dlt, &m_w_g2avg, &m_w_init, p, pa, do_reg); 
    do_reg = p.do_reg_intercept; 
    adam_update(&v_i, &v_i_grad, grad_num, &v_i_dlt, &v_i_g2avg, &v_i_init, p, pa, do_reg); 

    if (p.reg_L2const > 0) do_l2const(p);    
    grad_num = 0;    
  }

  /*---
   *  To ensure thread safety via the azc_thno/azc_thnum scheme, 
   *  AzPmat functions called here must go through matrix components 
   *  in the same order.  
   ---*/  
  void adam_update(AzPmat *m_weight, 
                       AzPmat *m_grad, /* input: grad */
                       int g_num, 
                       AzPmat *m_g1avg, 
                       AzPmat *m_g2avg, 
                       AzPmat *m_init, 
                       const AzpLmParam &p, 
                       const AzpLmAdam_Param &pa, 
                       bool do_reg) {
    ++tt;                          
                         
    double eta = pa.eta*pa.coeff; 

    m_grad->divide(-g_num);  /* negative gradient: -g_t */
    if (do_reg && !pa.do_shrink) add_reg_grad(p, 1, m_weight, m_grad, m_init); /* regularization */
    if (p.grad_clip > 0) m_grad->truncate(-p.grad_clip, p.grad_clip);     
    /*
     *  m_t = b1 m_{t-1} + (1-b1) g_t
     *  v_t = b2 v_{t-1} + (1-b2) g_t^2
     *  m'_t = m_t/(1-b1^t)
     *  v'_t = v_t/(1-b2^t) 
     *  theta_t = theta_{t-1} - alpha m'_t/(sqrt(v'_t)+eps)
     */
     
    m_g1avg->add(pa.b1, m_grad, 1-pa.b1); 
    m_g2avg->add_square(pa.b2, m_grad, 1-pa.b2);
    double b1t = pow(pa.b1, tt), b2t = pow(pa.b2, tt); 
    m_grad->adam_delta(m_g1avg, m_g2avg, b1t, b2t, pa.eps, pa.do_adam_var); 
    if (do_reg && pa.do_shrink) add_reg_grad(p, 1, m_weight, m_grad, m_init); /* shrink weights */    
    if (pa.grad_clip_after > 0) m_grad->truncate(-pa.grad_clip_after, pa.grad_clip_after); 
    m_weight->add(m_grad, eta); 
    if (p.weight_clip > 0) m_weight->truncate(-p.weight_clip, p.weight_clip); 
  }   
};
#endif 