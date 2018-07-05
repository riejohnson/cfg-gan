/* * * * *
 *  AzpMain_Cfg.hpp
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

#ifndef _AZP_MAIN_CFG_HPP_
#define _AZP_MAIN_CFG_HPP_

#include "AzpMain_reNet.hpp"
#include "AzpReNetG.hpp"
#include "AzpG_Cfg.hpp"
#include "AzpG_ClasEval.hpp"

class AzpMain_Cfg : public virtual AzpMain_reNet {
protected:
  AzOut less_out; 
  #define AzpCompoSetNum 10
  AzpCompoSetDflt g_cs[AzpCompoSetNum]; 
  /*---  overriding the one in AzpMain_reNet  ---*/
  virtual AzpReNet *alloc_renet_for_test(AzObjPtrArr<AzpReNet> &opa, AzParam &azp, int cs_idx) {  
    return alloc_renet(opa, azp, cs_idx, true); 
  }
  virtual AzpReNet *alloc_renet(AzObjPtrArr<AzpReNet> &opa, AzParam &azp, int cs_idx, bool for_test=false) {
    opa.alloc(1); 
    alloc_set_renet(opa, azp, cs_idx, for_test); 
    return opa(0); 
  } 
  virtual void alloc_set_renet(AzObjPtrArr<AzpReNet> &opa, 
                               AzParam &azp, int cs_idx, bool for_test=false) {
    AzX::throw_if(cs_idx<0 || cs_idx>=AzpCompoSetNum, "AzpMain_Cfg::alloc_set_renet", "cs_idx is out of range"); 
    for (int opa_idx = 0; opa_idx < opa.size(); ++opa_idx) {
      AzOut myout; if (opa_idx == 0) myout = less_out; 
      g_cs[cs_idx].reset(azp, for_test, myout); 
      opa.set(opa_idx, new AzpReNetG(&g_cs[cs_idx])); 
    }
  } 
  
public:
  AzpMain_Cfg() {} 
  void cfg_train(AzpG_Cfg &cfg, AzpG_ClasEval__ &cevs, int argc, const char *argv[], const AzBytArr &s_action);    
  void cfg_gen(AzpG_Gen &gen, int argc, const char *argv[], const AzBytArr &s_action);   
  void eval_img(AzpG_ClasEval__ &cevs, int argc, const char *argv[], const AzBytArr &s_action); 
  void gen_ppm(int argc, const char *argv[], const AzBytArr &s_action); 
  
protected:
  virtual void separate_network_param(int argc, const char *argv[], char dlm, 
                              AzBytArr &s_g_param, AzBytArr &s_c_param, AzBytArr &s_param) const;                                
  virtual AzpG_dg_info *read_mod(const AzBytArr &s_fn, AzpG_ddg &ddg, AzpG_dg_info &info) const; 
  virtual int read_num_in_mod(const AzBytArr &s_fn) const; 
}; 
#endif
