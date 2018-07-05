/* * * * *
 *  AzpBchNorm.hpp
 *  Copyright (C) 2018 Rie Johnson
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
#ifndef _AZP_BCH_NORM_HPP_
#define _AZP_BCH_NORM_HPP_

#include "AzPmat.hpp"
#include "AzpCompoSet_.hpp"

#define AzpReLayer_Type_BchNorm "BchNorm"

class AzpBchNorm_Param {
protected:   
  static const int version = 0; 
  static const int reserved_len = 32; 
public: 
  double bn_eps, bn_avg_coeff, bn_initw; 
  bool do_use_test_stat; 
  AzpBchNorm_Param() : bn_eps(1e-20), bn_initw(-1), 
                       bn_avg_coeff(0.75), do_use_test_stat(false) {}

  /*-----------------------------------------------*/  
  #define kw_bn_eps "bn_eps="
  #define kw_bn_avg_coeff "bn_avg_coeff="
  #define kw_bn_initw "bn_init_weight="
  #define kw_do_use_test_stat "BNuseTestStat"
  #define kw_do_use_test_stat_short "BNuts"
  
  /*-----------------------------------------------*/  
  virtual void resetParam(AzParam &azp, const char *pfx, bool is_warmstart) {
    azp.reset_prefix(pfx); 
    if (!is_warmstart) {
      azp.vFloat(kw_bn_initw, &bn_initw);        
      azp.vFloat(kw_bn_eps, &bn_eps);       
      azp.vFloat(kw_bn_avg_coeff, &bn_avg_coeff); 
      bn_avg_coeff = MIN(1, bn_avg_coeff);            
    }
    azp.swOn(&do_use_test_stat, kw_do_use_test_stat); 
    azp.swOn(&do_use_test_stat, kw_do_use_test_stat_short);       

    azp.reset_prefix(); 
  }
  virtual void checkParam(const char *pfx) {
    const char *eyec = "AzpBchNorm::checkParam"; 
    AzXi::throw_if_nonpositive(bn_eps, eyec, kw_bn_eps); 
    AzXi::throw_if_nonpositive(bn_avg_coeff, eyec, kw_bn_avg_coeff); 
  }
  virtual void printParam(const AzOut &out, const char *pfx) const {
    AzPrint o(out); 
    o.reset_prefix(pfx); 
    o.printV(kw_bn_initw, bn_initw); 
    o.printV(kw_bn_eps, bn_eps);     
    o.printV(kw_bn_avg_coeff, bn_avg_coeff);     
    o.printSw(kw_do_use_test_stat, do_use_test_stat); 
    o.printEnd(); 
  } 
  virtual void printHelp(AzHelp &h) const {}
  virtual void write(AzFile *file) const {
    AzTools::write_header(file, version, reserved_len);  
    file->writeDouble(bn_eps);   
    file->writeDouble(bn_avg_coeff); 
    file->writeBool(false); /* for compatibility */
    file->writeBool(do_use_test_stat); 
    file->writeBool(false); /* for compatibility */
    file->writeBool(false); /* for compatibility */
  }
  virtual void read(AzFile *file) {
    AzTools::read_header(file, reserved_len);   
    bn_eps = file->readDouble(); 
    bn_avg_coeff = file->readDouble(); 
    bool dummy0 = file->readBool(); /* for compatibility */
    do_use_test_stat = file->readBool(); 
    bool dummy1 = file->readBool(); /* for compatibility */
    bool dummy2 = file->readBool(); /* for compatibility */
  }   
  virtual void resetParam(const AzOut &out, AzParam &azp, const AzPfx &pfx, bool is_warmstart=false) {
    for (int px=0; px<pfx.size(); ++px) resetParam(azp, pfx[px], is_warmstart); 
    checkParam(pfx.pfx()); 
    printParam(out, pfx.pfx()); 
  }   
}; 
  
class AzpBchNorm {
protected:
  AzpBchNorm_Param p; 
  AzPmat v_mu, v_si; /* mu: average, sigma: sqrt(var+eps) */
  AzPmat m_x, m_xhat, m_xld_sv; 
  AzPmat v_mu_avg, v_si_avg, v_var_avg; 

  AzObjPtrArr<AzpWeight_> wei;

  virtual void init_wei(const AzpLayerCompoPtrs &cs, int cc, const AzOut &out, AzParam &azp, 
                        const AzPfx &pfx, bool is_warmstart, bool for_testonly) {
    if (!is_warmstart) {                          
      wei.free_alloc(2); 
      for (int ix = 0; ix < 2; ++ix) wei.set(ix, cs.weight->clone());
    }
    for (int ix = 0; ix < 2; ++ix) {
      AzpWeight_ *wp = wei(ix); 
      if (!for_testonly) wp->resetParam(azp, pfx, is_warmstart); /* no print */    
      wp->force_no_intercept(); 
      wp->force_no_reg(); 
    }                         
    if (!is_warmstart) {
      for (int ix = 0; ix < 2; ++ix) wei(ix)->reset(-1, 1, cc, false, true);       
      AzPmat v_ga(ga()), v_be(be()); 
      v_ga.set(1); v_be.set((double)0); 
      if (p.bn_initw > 0) {
        AzDmat mg(v_ga.rowNum(), v_ga.colNum()); 
        AzRandGen rg; rg.gaussian(p.bn_initw, &mg); 
        v_ga.add(&mg); 
      }
      reset_ga_be(v_ga, v_be);             
    }
  }
  virtual const AzPmat *ga() const { return wei[0]->linmod()->weights(); }
  virtual const AzPmat *be() const { return wei[1]->linmod()->weights(); }
  virtual void reset_ga_be(const AzPmat &v_ga, const AzPmat &v_be) {
    AzPmat v0(1, 1); 
    AzpLm_Untrainable lm_ga(&v_ga, &v0); wei(0)->initWeights(&lm_ga, 1); 
    AzpLm_Untrainable lm_be(&v_be, &v0); wei(1)->initWeights(&lm_be, 1);    
  }
  virtual void updateDelta_ga_be(int data_num, const AzPmat &v_gld, const AzPmat *v_bld=NULL) {
    wei(0)->updateDelta(data_num, (AzPmat *)NULL, &v_gld); 
    if (v_bld != NULL) wei(1)->updateDelta(data_num, (AzPmat *)NULL, v_bld); 
  } 
  virtual void flushDelta_ga_be() {
    for (int ix = 0; ix < 2; ++ix) { 
      wei(ix)->flushDelta(); 
      wei(ix)->linmod_u()->flush_ws(); /* without this, ga() and be() cause error when there is no mememtum */
    }
  }   
  virtual void _multiply_to_stepsize(double factor) { for (int ix = 0; ix < 2; ++ix) wei(ix)->multiply_to_stepsize(factor); }
  /*---------------------------------------------------------------------------------------------*/   
  
  static const int version = 0; 
  static const int reserved_len = 32;  
public:
  AzpBchNorm() {}
  const char *name() const { return AzpReLayer_Type_BchNorm; }
  virtual void reset(const AzpLayerCompoPtrs &cs, int cc, const AzOut &out, AzParam &azp, 
                     const AzPfx &pfx, bool is_warmstart=false, bool for_testonly=false) {
    p.resetParam(out, azp, pfx, is_warmstart);    
    v_mu.reform(cc, 1); 
    v_si.reform(cc, 1); v_si.set(1); 
    init_wei(cs, cc, out, azp, pfx, is_warmstart, for_testonly); 
  }  
  virtual void printHelp(AzHelp &h) const { p.printHelp(h); }  
  virtual void write(AzFile *file) const {
    AzTools::write_header(file, version, reserved_len);  
    p.write(file); 
    v_mu_avg.write(file); v_var_avg.write(file); 
    wei[0]->write(file); wei[1]->write(file);    
  }
  virtual void read(const AzpLayerCompoPtrs &cs, AzFile *file) {
    AzTools::read_header(file, reserved_len);   
    p.read(file); 
    v_mu_avg.read(file); v_var_avg.read(file); 
    wei.free_alloc(2); 
    for (int ix = 0; ix < 2; ++ix) wei.set(ix, cs.weight->clone());
    wei(0)->read(file); wei(1)->read(file);         
  }    
 
  virtual void multiply_to_stepsize(double factor) { _multiply_to_stepsize(factor); }
  virtual void show_stat(AzBytArr &s) const {
    const AzPmat *v_ga = ga(), *v_be = be(); 
    s << "bn:gamma_avg," << v_ga->sum()/(double)v_ga->size() << "," << "beta_avg," << v_be->sum()/(double)v_be->size() << ","; 
    s << "gamma_absavg," << v_ga->absSum()/(double)v_ga->size() << ",beta_absavg," << v_be->absSum()/(double)v_be->size() << ","; 
    s << "muavg," << v_mu.sum()/(double)v_mu.size() << "," << "siavg," << v_si.sum()/(double)v_si.size() << ","; 
    s << "muabsavg," << v_mu.absSum()/(double)v_mu.size() << ",siabsavg," << v_si.absSum()/(double)v_si.size() << ",";     
  }
  
  /*---  upward  ---*/
  virtual void upward(bool is_test, AzPmat &m_inout) { AzPmat m; upward(is_test, m_inout, m); m_inout.set(&m); }    
  virtual void upward(bool is_test, const AzPmat &m_inp, AzPmat &m_out) {
    /*---  prepare average and standard deviation  ----*/
    if (is_test) {
      if (p.do_use_test_stat)         avg_var_sdev(m_inp, v_mu, NULL, v_si, p.bn_eps); 
      else if (is_there_moving_avg()) get_moving_avg(v_mu, v_si); 
      else                            avg_var_sdev(m_inp, v_mu, NULL, v_si, p.bn_eps); 
    }
    else {
      m_x.set(&m_inp);       
      AzPmat v_var; avg_var_sdev(m_inp, v_mu, &v_var, v_si, p.bn_eps); 
      if (!p.do_use_test_stat) update_moving_avg(v_mu, v_var); 
    }

    /*---  compute ...  ---*/
    m_xhat.set_subcol_divcol(m_inp, v_mu, v_si); 
    m_out.set_mulcol_addcol(m_xhat, *ga(), *be()); 
  }
  
  /*---  downward  ---*/
  virtual void downward(AzPmat &m_inout, int data_num, bool dont_update,  bool dont_release_sv) { 
    AzPmat m; _downward(m_inout, m, data_num, dont_update, dont_release_sv); 
    m_inout.set(&m);    
  }  
  /* "p:" abbreviates "partial " */
  /* input: p:L / p:y_i, output: p:L / p:x_i */
  virtual void _downward(const AzPmat &m_yld, AzPmat &m_xld, int data_num, bool dont_update, bool dont_release_sv) {  
    int num = m_yld.colNum(); 
    AzX::throw_if(m_x.colNum() != num || m_xhat.colNum() != num, "AzpBchNorm::downward", 
                  "No input data saved (m_x or m_xhat)"); 

    /*---  m_xhatld <- p:L/p:xhat --- */
    AzPmat m_xhatld; m_xhatld.set_mulcol(m_yld, *ga()); 

    /*---  m_xld <- p:L/p:x  ---*/  
    /*---  p:L/p:x = s[p:L/p:xhat - mean(p:L/p:xhat) - mean(p:L/p:xhat xhat) xhat] ---*/ 
    AzPmat m0; m0.avg_per_row(&m_xhatld); /* m0 <- mean(p:L/p:xhat) */
    AzPmat m1; m1.set_mul(m_xhatld, m_xhat); 
    AzPmat m2; m2.avg_per_row(&m1); /* m2 <- mean(p:L/p:xhat xhat) */
    m1.set_mulcol(m_xhat, m2); 
    m_xld.set_subcol_sub_divcol(m_xhatld, m0, m1, v_si); 
    
    m_xld_sv.set(&m_xld); /* saving for upward2 */
    
    if (!dont_update) {
      m0.set(&m_yld); m0.elm_multi(&m_xhat); 
      AzPmat vg; vg.sum_per_row(&m0);  /* p:L/p:gamma */
      AzPmat vb; vb.sum_per_row(&m_yld);  /* p:L/p:beta */
      updateDelta_ga_be(data_num, vg, &vb);     
    }
    if (!dont_release_sv) release_sv(); 
  }

  virtual void release_sv() { m_x.destroy(); m_xhat.destroy(); }
  virtual void release_ld() {}

  /*---  upward2  ---*/
  virtual void upward2(const AzPmat &m_revx_ld, /* p:L'/p:x'_k */
                       AzPmat &m_revy_ld, int data_num, bool dont_update) {
    int num = m_revx_ld.colNum(); 
    AzX::throw_if(m_xhat.colNum() != num, "AzpBchNorm::upward2", "No xhat saved"); 

    /*---  p:L'/p:y'_k = s gamma[p:L'/p:x'_k - mean(p:L'/p:x') - mean(p:L'/p:x' xhat) xhat_k] ---*/
    AzPmat m0; m0.avg_per_row(&m_revx_ld);        /* m0 <- mean(p:L'/p:x') */
    AzPmat m1(&m_revx_ld); m1.elm_multi(&m_xhat); /* m1 <- p:L'/p:x' xhat */
    AzPmat m2; m2.avg_per_row(&m1);               /* m2 <- mean(p:L'/p:x' xhat) */
    m1.set(&m_xhat); m1.multi_col(&m2);           /* m1 <- mean(p:L'/p:x' xhat) xhat */
    m_revy_ld.set(&m_revx_ld); m_revy_ld.add_col(&m0, -1); m_revy_ld.add(&m1, -1); 
    m_revy_ld.divide_col(&v_si); m_revy_ld.multi_col(ga());    
    
    if (!dont_update) {
      /*---  p:L/p:gamma = (1/gamma) sum(p:L'/p:x' x')  ---*/
      m_xld_sv.elm_multi(&m_revx_ld); 
      AzPmat vg; vg.sum_per_row(&m_xld_sv); vg.elm_divide(ga()); 
      updateDelta_ga_be(data_num, vg);       
      m_xld_sv.destroy(); 
    }
  }
   
  virtual void flushDelta() { flushDelta_ga_be(); }

protected:  
  static void avg_var_sdev(const AzPmat &m_x, AzPmat &va, AzPmat *ovv, AzPmat &vs, double eps) {
    /*---  compute average and variance */
    va.avg_per_row(&m_x); /* average */
    AzPmat m0; m0.set_square(m_x); vs.avg_per_row(&m0); /* vs <- mean(x^2) */
    AzPmat va2; va2.set_square(va); /* va2 <- (mean(x))^2 */ 
    vs.add(&va2, -1); /* vs <- mean(x^2)-(mean(x))^2 */
    if (ovv != NULL) ovv->set(&vs); 
    vs.add(eps); vs.squareroot(); /* vs <- sqrt( mean(x^2)-(mean(x))^2 )+eps ) */
  }  
  virtual bool is_there_moving_avg() const { return (v_mu_avg.size() > 0); }
  virtual void update_moving_avg(const AzPmat &v_mu, const AzPmat &v_var) {
    if (v_mu_avg.size() <= 0) {
      v_mu_avg.set(&v_mu); v_var_avg.set(&v_var);  v_si_avg.reset(); 
    }
    else {
      v_mu_avg.add(1-p.bn_avg_coeff, &v_mu, p.bn_avg_coeff); 
      v_var_avg.add(1-p.bn_avg_coeff, &v_var, p.bn_avg_coeff);
      v_si_avg.reset(); 
    }
  }
  virtual void get_moving_avg(AzPmat &v_mu, AzPmat &v_si) { 
    v_mu.set(&v_mu_avg); 
    if (v_si_avg.size() <= 0) {
      v_si_avg.set(&v_var_avg); v_si_avg.add(p.bn_eps); v_si_avg.squareroot();
    }
    v_si.set(&v_si_avg);   
  }    
}; 
#endif 
