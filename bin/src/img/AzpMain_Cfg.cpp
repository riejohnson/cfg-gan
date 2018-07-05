/* * * * *
 *  AzpMain_Cfg.cpp
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

#include "AzpMain_Cfg.hpp"
#include "AzTools.hpp"

extern AzByte param_dlm; 

/*------------------------------------------------------------*/ 
#define kw_eval_fn "evaluation_fn="
#define kw_warmfn "fn_for_warmstart="
#define kw_do_use_y "Y"
#define kw_cfg_T "cfg_T=" 
#define kw_do_verbose "Verbose"
/*------------------------------------------------------------*/
#define g_pfx "_g_"
#define d_pfx "_d_"
class AzpMain_Cfg_train_Param : public virtual AzpMain_reNet_Param_ {
public:
  int cfg_T; 
  AzpDataSetDflt r_ds; /* real data */
  AzBytArr s_eval_fn, s_warmfn; 
  bool do_use_y, do_verbose; 
  AzpMain_Cfg_train_Param(AzParam &azp, const AzOut &out, const AzBytArr &s_action) 
    : cfg_T(-1), do_use_y(false), do_verbose(false) {
    reset(azp, out, s_action); 
  } 
  void resetParam(const AzOut &out, AzParam &p) {
    AzPrint o(out); 
    _resetParam(o, p); 

    p.swOn(o, do_use_y, kw_do_use_y); 
    
    AzBytArr s_tstnm; p.vStr(kw_tstnm, &s_tstnm); 
    bool do_train = true, do_test = (s_tstnm.length() > 0); 
    r_ds.resetParam(out, p, do_train, do_test, do_use_y);
    p.vStr_prt_if_not_empty(o, kw_eval_fn, s_eval_fn); 
    p.vStr_prt_if_not_empty(o, kw_warmfn, s_warmfn); 
    p.vInt(o, kw_cfg_T, cfg_T);        
    p.swOn(&do_verbose, kw_do_verbose); 
    setupLogDmp(o, p);   
  } 
}; 
 
/*------------------------------------------------------------*/
void AzpMain_Cfg::separate_network_param(int argc, const char *argv[], char dlm, 
                                 AzBytArr &s_g_param, /* approximator/generator */
                                 AzBytArr &s_d_param, /* discriminator */
                                 AzBytArr &s_param) const {
  AzBytArr s_all;                                    
  int glen = (int)strlen(g_pfx), clen = (int)strlen(d_pfx);     
  for (int ix = 0; ix < argc; ++ix) {
    const char *pt = argv[ix]; 
    if (*pt == '@') {
      AzFile file(pt+1); file.open("rb"); int fsz = (int)file.size();       
      AzBytArr s_buf; AzByte *buf = s_buf.reset(fsz, 0); 
      file.readBytes(buf, fsz); file.close(); 
      AzStrPool sp; AzTools::getStrings(buf, fsz, &sp); 
      for (int tx = 0; tx < sp.size(); ++tx) {
        const char *tok = sp.c_str(tx); 
        if      (AzBytArr::beginsWith(tok, g_pfx)) s_g_param << tok+glen << dlm; 
        else if (AzBytArr::beginsWith(tok, d_pfx)) s_d_param << tok+clen << dlm; 
        else                                       s_param << tok << dlm;           
      }
    }
    if      (AzBytArr::beginsWith(pt, g_pfx)) s_g_param << pt+glen << dlm; 
    else if (AzBytArr::beginsWith(pt, d_pfx)) s_d_param << pt+clen << dlm; 
    else                                      s_param << pt << dlm; 
    s_all << pt << dlm; 
  }   
  AzPrint::writeln(log_out, s_all); 
  AzPrint::writeln(log_out, "--------------------");    
}

/*------------------------------------------------------------*/
void AzpMain_Cfg::cfg_train(AzpG_Cfg &cfg, AzpG_ClasEval__ &cevs,
                               int argc, const char *argv[], const AzBytArr &s_action) {
  const char *eyec = "AzpMain_Cfg::cfg_train";
  
  AzBytArr s_g_param, s_d_param, s_param;
  separate_network_param(argc, argv, param_dlm, s_g_param, s_d_param, s_param);
  AzParam g_azp(s_g_param.c_str(), true, param_dlm);
  AzParam d_azp(s_d_param.c_str(), true, param_dlm);
  AzParam azp(s_param.c_str(), true, param_dlm);
  AzpMain_Cfg_train_Param p(azp, log_out, s_action);
  if (p.do_verbose) less_out = log_out;
  
  /*---  read training data  ---*/
  int dummy_ydim = (p.do_use_y) ? -1 : 1;
  p.r_ds.reset_data(log_out, dummy_ydim);  /* real data */
 
  /*---  allocate areas for networks  ---*/
  int num_in_file = read_num_in_mod(p.s_warmfn);
  int D_num = MAX(p.cfg_T, num_in_file);
  AzPrint::writeln(log_out, "Number of allocated D's = ", D_num);
  AzObjPtrArr<AzpReNet> arr_d_net(D_num), opa_g, opa_cls; /* owning networks */
  int d_cs_idx=0, g_cs_idx=1, cls_cs_idx=2;
  alloc_set_renet(arr_d_net, d_azp, d_cs_idx);
  AzpReNet *g_net = alloc_renet(opa_g, g_azp, g_cs_idx);
  AzpG_ddg ddg(g_net, arr_d_net);
  AzpG_dg_info info;
  AzpG_dg_info *infp = read_mod(p.s_warmfn, ddg, info);
  
  /*---  Set up for evaluation using a classifier  ---*/ 
  AzParam clas_azp("", true, param_dlm);  
  AzpReNet *clas_net_tmpl = alloc_renet_for_test(opa_cls, clas_azp, cls_cs_idx);
  cevs.reset(less_out, azp, clas_net_tmpl);

  /*---  For writing the evaluation results  ---*/
  AzOut eval_out; AzOfs ofs;
  AzOut *eval_out_ptr = reset_eval(p.s_eval_fn, ofs, eval_out);
  
  /*---  xICFG  ---*/
  AzClock clk;
  cfg.cfg_train(eval_out_ptr, ddg, infp, p.r_ds, g_azp, d_azp, azp, cevs);
  AzTimeLog::print("Done ...", log_out);
  clk.tick(log_out, "elapsed: ");
  if (ofs.is_open()) ofs.close();
}

/*------------------------------------------------------------*/   
 AzpG_dg_info *AzpMain_Cfg::read_mod(const AzBytArr &s_fn, 
                   AzpG_ddg &ddg, AzpG_dg_info &info) const {
  AzpG_dg_info *infp = NULL;                      
  if (s_fn.length() > 0) {
    AzX::throw_if(!s_fn.endsWith(".ddg"), "AzpMain_Cfg::read_mod", "Unknown file extension.  Expected *.ddg."); 
    AzTimeLog::print("Reading ", s_fn.c_str(), log_out); 
    ddg.read_nets(s_fn.c_str(), info); 
    infp = &info; 
  }
  return infp; 
}  
/*------------------------------------------------------------*/ 
int AzpMain_Cfg::read_num_in_mod(const AzBytArr &s_fn) const {
  int num = 0; 
  if (s_fn.length() > 0 && s_fn.endsWith(".ddg")) {
    num = AzpG_ddg::read_num_in_file(s_fn.c_str()); 
  }
  return num; 
}
  
/*------------------------------------------------------------*/    
/*------------------------------------------------------------*/  
class AzpMain_Cfg_gen_Param : public virtual AzpMain_reNet_Param_ {
public:
  bool do_verbose; 
  AzBytArr s_mod_fn; 
  AzpMain_Cfg_gen_Param(AzParam &azp, const AzOut &out, const AzBytArr &s_action) 
    : do_verbose(false) {
    reset(azp, out, s_action); 
  }
  #define kw_mod_fn "model_fn="
  void resetParam(const AzOut &out, AzParam &p) {
    const char *eyec = "AzpMain_Cfg_gen_Param::resetParam";    
    AzPrint o(out); 
    _resetParam(o, p);          
    p.vStr(o, kw_mod_fn, s_mod_fn); 
    AzXi::throw_if_empty(s_mod_fn, eyec, kw_mod_fn); 
    p.swOn(o, do_verbose, kw_do_verbose); 
  }   
}; 

/*------------------------------------------------------------*/
void AzpMain_Cfg::cfg_gen(AzpG_Gen &gen, int argc, const char *argv[], const AzBytArr &s_action) {
  const char *eyec = "AzpMain_Cfg::cfg_gen"; 
  AzBytArr s_g_param, s_d_param, s_param; 
  separate_network_param(argc, argv, param_dlm, s_g_param, s_d_param, s_param);                
  AzParam g_azp(s_g_param.c_str(), true, param_dlm);
  AzParam d_azp(s_d_param.c_str(), true, param_dlm);
  AzParam azp(s_param.c_str(), true, param_dlm);
  AzpMain_Cfg_gen_Param p(azp, log_out, s_action);
  if (p.do_verbose) less_out = log_out;
  
  /*---  allocate networks  ---*/
  int D_num = 0;
  if (p.s_mod_fn.endsWith(".ddg")) { /* gan */
    D_num = AzpG_ddg::read_num_in_file(p.s_mod_fn.c_str());
  }
  AzPrint::writeln(log_out, "Number of allocated D's = ", D_num);
  
  AzObjPtrArr<AzpReNet> arr_d_net(D_num), opa_g; /* owning networks */
  int d_cs_idx=0, g_cs_idx=1;
  alloc_set_renet(arr_d_net, d_azp, d_cs_idx);
  AzpReNet *g_net = alloc_renet(opa_g, g_azp, g_cs_idx);
  AzpG_ddg ddg(g_net, arr_d_net);

  /*---  read the model  ---*/
  AzpG_dg_info info; 
  AzTimeLog::print("Reading ", p.s_mod_fn.c_str(), log_out);
  if (p.s_mod_fn.endsWith(".ddg")) {  /* cfg */
    ddg.read_nets(p.s_mod_fn.c_str(), info);
  }
  else AzX::throw_if(true, AzInputError, eyec,
       "The model filename should end with \".ddg\".");

  /*---  generate  ---*/
  gen.generate(ddg, info, g_azp, d_azp, azp);
  AzTimeLog::print("Done ...", log_out);
}

/*------------------------------------------------------------*/    
#define kw_minib "mini_batch_size="
#define kw_seed "random_seed="
#define kw_num "num="
/*------------------------------------------------------------*/  
class AzpMain_Cfg_eval_img_Param : public virtual AzpMain_reNet_Param_ {
public:
  int num, seed; 
  AzpDataSetDflt ds; 
  int minib; 
  bool do_verbose; 
  AzpMain_Cfg_eval_img_Param(AzParam &azp, const AzOut &out, const AzBytArr &s_action) 
     : minib(64), num(-1), seed(-1), do_verbose(false) {
    reset(azp, out, s_action); 
  }
  void resetParam(const AzOut &out, AzParam &p) {
    const char *eyec = "AzpMain_Cfg_eval_img_Param::resetParam"; 
    AzPrint o(out); 
    
    bool do_train=false, do_test=true, is_there_y=false; 
    ds.resetParam(out, p, do_train, do_test, is_there_y);    
    
    p.vInt(o, kw_seed, seed); 
    if (seed > 0) p.vInt(o, kw_num, num); 
    p.vInt(o, kw_minib, minib);
    AzXi::throw_if_nonpositive(minib, eyec, kw_minib); 
    p.swOn(o, do_verbose, kw_do_verbose); 
  }   
}; 

/*------------------------------------------------------------*/
void AzpMain_Cfg::eval_img(AzpG_ClasEval__ &cevs, int argc, const char *argv[], const AzBytArr &s_action) {
  const char *eyec = "AzpMain_Cfg::eval_img";
  AzParam azp(param_dlm, argc, argv);
  AzpMain_Cfg_eval_img_Param p(azp, log_out, s_action);
  if (p.do_verbose) less_out = log_out;
  
  /*---  Read image data  ---*/
  int dummy_ydim = 1;
  p.ds.reset_data(log_out, dummy_ydim);
  
  /*---  Set up for evaluation using a classifier  ---*/
  AzParam clas_azp("", true, param_dlm);
  AzObjPtrArr<AzpReNet> opa_cls;
  int cls_cs_idx = 0;
  AzpReNet *clas_net_tmpl = alloc_renet_for_test(opa_cls, clas_azp, cls_cs_idx);
  cevs.reset(less_out, azp, clas_net_tmpl);
  AzX::throw_if(cevs.size()<=0, AzInputError, eyec,
                "\"classif_fn=\"", "must be specified.");
  azp.check(log_out);
  
  /*---  Evaluate the images  ---*/
  if (p.seed > 0 && p.num > 0) {
    AzRandGen rg; rg._srand_(p.seed);
    AzIntArr ia_dxs; rg._sample_(p.ds.tst_data()->dataNum(), p.num, ia_dxs);
    AzpG_ClasEval_::eval_img(log_out, p.do_verbose, cevs, p.ds.tst_data(), p.minib, &ia_dxs);
  }
  else {
    AzpG_ClasEval_::eval_img(log_out, p.do_verbose, cevs, p.ds.tst_data(), p.minib);
  }
}

/*------------------------------------------------------------*/
#define kw_seed "random_seed="
#define kw_gen_fn "gen_fn="
#define kw_gen_num "num_gen="
#define kw_ww "w="
#define kw_hh "h="
#define kw_gap "gap="
#define kw_cls_no "class="
#define kw_do_entropy "Entropy"
#define kw_each_num "num_each="

#define kw_do_rev "Rev"
#define kw_first "first="
#define kw_interval "interval="
#define kw_do_compati "Compati"
#include "AzpData_imgbin.hpp" /* for kw_do_pm1 */
#include "AzpG_Tools.hpp"
/*------------------------------------------------------------*/
class AzpMain_Cfg_gen_ppm_Param : public virtual AzpMain_reNet_Param_ {
public:
  AzpDataSetDflt ds;
  bool do_pm1, do_verbose;
  int seed, gen_num, minib;
  int ww, hh;  /* # of images horizaontally and vertically to make a collage */
  int gap; /* gap between images to make a collage */
  AzBytArr s_gen_fn, s_clas_fn;
  bool do_entropy, do_rev, do_compati;
  int cls_no, first, interval, each_num;
  AzpMain_Cfg_gen_ppm_Param(AzParam &azp, const AzOut &out, const AzBytArr &s_action)
    : seed(-1), gen_num(-1), minib(64), ww(-1), hh(-1), gap(-1), do_entropy(false), do_verbose(false),
      do_rev(false), first(-1), interval(-1), do_compati(false), cls_no(-1), each_num(-1) {
    reset(azp, out, s_action);
  }

  bool doing_collage() const { return (ww > 0); }
  
  void resetParam(const AzOut &out, AzParam &p) {
    const char *eyec = "AzpMain_Cfg_gen_ppm_Param::resetParam";
    AzPrint o(out);
    _resetParam(o, p);

    bool do_train=false, do_test=true, is_there_y=false;
    ds.resetParam(out, p, do_train, do_test, is_there_y);
    p.vInt(o, kw_seed, seed);
    p.vStr(o, kw_gen_fn, s_gen_fn);
    p.vInt(o, kw_gen_num, gen_num);
    p.vInt(o, kw_ww, ww);
    if (doing_collage()) {
      p.vStr(o, kw_clas_fn, s_clas_fn);
      if (s_clas_fn.length() > 0) {    
        p.swOn(o, do_entropy, kw_do_entropy);
        if (!do_entropy) {
          p.vInt(o, kw_cls_no, cls_no);
          if (cls_no < 0) p.vInt(o, kw_each_num, each_num);
        }
        if (cls_no >= 0 || do_entropy || each_num > 0) p.swOn(o, do_rev, kw_do_rev);
      }
      p.vInt(o, kw_hh, hh);
      p.vInt(o, kw_gap, gap);
      AzXi::throw_if_nonpositive(hh, eyec, kw_hh);
      gap = MAX(gap, 0);
      if (gen_num <= 0) gen_num = ww*hh;
      AzX::throw_if(gen_num<ww*hh, eyec, "Conflict between (w,h) and num_gen");
      p.vInt(o, kw_first, first);
      if (first >= 0) {
        interval = 1; /* default */
        p.vInt(o, kw_interval, interval);
        int num = DIVUP(gen_num-first, interval);
        AzX::throw_if(num<ww*hh, eyec, "Conflict between (w,h) and (num_gen,first,interval)");
      }
    }
    else {
      AzXi::throw_if_nonpositive(gen_num, eyec, kw_gen_num);
    }
    p.swOn(o, do_pm1, kw_do_pm1);
    p.vInt(o, kw_minib, minib);
    p.swOn(o, do_verbose, kw_do_verbose);
    p.swOn(o, do_compati, kw_do_compati);
    AzXi::throw_if_empty(s_gen_fn, eyec, kw_gen_fn);
    AzXi::throw_if_nonpositive(minib, eyec, kw_minib);
    setupLogDmp(o, p);
  }
};

/*------------------------------------------------------------*/
void AzpMain_Cfg::gen_ppm(int argc, const char *argv[], const AzBytArr &s_action) {
  const char *eyec = "AzpMain_Cfg::gen_ppm";

  AzParam azp(param_dlm, argc, argv);
  AzpMain_Cfg_gen_ppm_Param p(azp, log_out, s_action);
  azp.check(log_out);

  /*---  Read a dataset  ---*/
  int dummy_ydim = 1;
  p.ds.reset_data(log_out, dummy_ydim);
  const AzpData_ *data = p.ds.tst_data();
  AzX::throw_if_null(data, eyec, "data (pointer to the dataset)");

  /*---  sample data  ---*/
  AzIntArr ia_dxs;
  if (p.seed > 0) {
    srand(p.seed);
    AzRandGen rg; 
    if (p.do_compati) rg._srand_(p.seed);
    rg._sample_(data->dataNum(), p.gen_num, ia_dxs);
  }
  else {
    ia_dxs.range(0, MIN(data->dataNum(), p.gen_num));
  }

  if (p.doing_collage()) { /* generate a collage ppm */
    /*---  load a classifier if specified  ---*/
    AzpReNet *clas_net = NULL; AzObjPtrArr<AzpReNet> opa_cls;
    if (p.s_clas_fn.length() > 0) {
      AzTimeLog::print("Reading ", p.s_clas_fn.c_str(), log_out);
      int cls_cs_idx = 0;
      AzParam clas_azp("", true, param_dlm);
      clas_net = alloc_renet_for_test(opa_cls, clas_azp, cls_cs_idx);
      if (!p.do_verbose) clas_net->deactivate_out();
      clas_net->read(p.s_clas_fn.c_str());
      AzTimeLog::print("Sorting images by predicted classes ... ", log_out);
      AzpG_Tools::order_by_cls(clas_net, p.minib, data, ia_dxs,
                               p.do_entropy, p.cls_no, p.do_rev, p.each_num);
      if (p.first >= 0 && p.interval > 0) {
        AzIntArr ia(&ia_dxs); ia_dxs.reset();
        for (int ix = p.first; ix < ia.size(); ix += p.interval) ia_dxs.put(ia[ix]);
      }
      ia_dxs.cut(p.ww*p.hh);
    }
    AzpG_Tools::gen_collage_ppm(data, p.do_pm1, ia_dxs, p.gap, p.s_gen_fn.c_str(),
                                p.ww, p.hh);
  }
  else { /* generate ppm files */
    int width = data->size(0), height = data->size(1), cc = data->xdim();
    for (int ix = 0; ix < ia_dxs.size(); ix += p.minib) {
      int d_num = MIN(p.minib, ia_dxs.size()-ix);
      AzPmatVar mv_data; data->gen_data(ia_dxs.point()+ix, d_num, mv_data);
      AzpG_Tools::to_img_ppm(mv_data, width, height, cc, p.do_pm1, p.s_gen_fn.c_str(), ix);
    }
  }
  AzTimeLog::print("Done ... ", log_out);
}
