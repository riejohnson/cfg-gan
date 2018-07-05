/* * * * *
 *  AzpG_ClasEval.hpp
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
#ifndef _AZP_G_CLAS_EVAL_HPP_
#define _AZP_G_CLAS_EVAL_HPP_
 
#include "AzpReNet.hpp" 

class AzpG_ClasEval_ {
public:
  virtual void init(const AzOut &out, const AzpData_ *r_trn, const AzpData_ *r_tst, int minib) = 0; 
  virtual void eval_init(int num) = 0; 
  virtual void eval_clas(const AzPmatVar &mv_gen, int dx_begin, int d_num) = 0; 
  virtual void show_eval(AzBytArr &s, const char *pfx="") = 0; 
  
  /*---  ---*/
  static void eval_img(const AzOut &out, bool do_verbose, 
                      AzpG_ClasEval_ &cevs, const AzpData_ *tst, int mb, 
                      const AzIntArr *_ia_dxs=NULL, const AzpData_ *data_for_init=NULL);                        
};

class AzpG_ClasEval__ : public virtual AzpG_ClasEval_ { /* for setting up the evaluator */
public:
  virtual void reset(const AzOut &out, AzParam &azp, const AzpReNet *clas_net_tmpl) = 0;   
  virtual int size() const = 0; /* # of classifiers */
}; 

/*----------------------------------------------------------------------*/
/* Evaluator using a classifier */
class AzpG_ClasEval : public virtual AzpG_ClasEval_ {
protected:
  AzOut out; 
  AzObjPtrArr<AzpReNet> arr; /* own it */
  AzpReNet *clas_net;  /* just pointing */
  int myno; 
  
  bool do_verbose; 

  /*---  for the score  ---*/
  AzBytArr s_snm; 
  AzPmat m_out;  
  
  /*---  for class dist  ---*/
  AzBytArr s_cnm; 
  AzBytArr s_y_fn;
  AzPmat v_real_y;   
public:  
  AzpG_ClasEval() : clas_net(NULL), myno(-1), do_verbose(false), 
                    s_snm("score"), s_cnm("classKL") {}

  virtual void reset() {
    v_real_y.reset(); m_out.reset();     
    arr.free(); clas_net = NULL;    
  }
  virtual void reset_clas_net(const AzOut &_out, const char *fn, const AzpReNet *clas_net_tmpl) {
    out = _out; 
    reset(); 
    AzX::throw_if_null(clas_net_tmpl, "AzpClasEval::reset_clas_net", "template");  
    clas_net = clas_net_tmpl->safe_clone_nocopy(arr); 
    AzTimeLog::print("Reading ", fn, out);   
    clas_net->read(fn); 
  }   

  virtual void reset_y_fn(const char *y_fn) {   
    s_y_fn.reset(y_fn);  
    v_real_y.reset(); m_out.reset();     
  } 
  virtual void init(const AzOut &out, const AzpData_ *r_trn, const AzpData_ *r_tst, int minib) {  
    init(0, out, r_trn, r_tst, minib); 
  }
  virtual void init(int no, const AzOut &_out, const AzpData_ *r_trn, 
                    const AzpData_ *r_tst, /* not used */
                    int minib) {
    out = _out; 
    myno = no; 
    if (clas_net == NULL) return; 
    AzX::throw_if_null(r_trn, "AzpG_ClasEval::init"); 
    
    /*---  for class dist  ---*/
    if (s_y_fn.length() > 0) _init_real_y(out, s_y_fn.c_str());  
    
    /*---  initialize the classifier  ---*/    
    AzPrint::writeln(out, "=====  clas_net  ====="); 
    if (out.isNull()) clas_net->deactivate_out(); 
    AzParam clas_azp(""); 
    clas_net->init_test(clas_azp, r_trn); 
  }
  virtual void eval_init(int num) {
    if (clas_net == NULL) return; 
    m_out.reform(clas_net->classNum(), num);  
  }
  virtual void eval_clas(const AzPmatVar &mv_gen, int dx_begin, int d_num); 
  virtual void show_eval(AzBytArr &s, const char *pfx=""); 
  void resetParam(int no, const AzOut &out, AzParam &azp) {}  
  
protected:  
  virtual void _show_scores(AzBytArr &s, const AzBytArr &ss); 
  virtual void _init_real_y(const AzOut &out, const char *y_fn); 
  virtual double _get_kl(const AzPmat &_v0, const AzPmat &_v1) const; 
  static double _eval_score(const AzPmat &m_out, AzPmat *v_sclas=NULL);  
  static void _smooth(AzPmat &m, double eps);    
  template <class T> static void name_value(AzBytArr &s, const AzBytArr &ss, const char *nm, T v) {
    s << "," << nm<<ss << "," << v; 
  }
  template <class T> static void name_value(AzBytArr &s, const AzBytArr &ss, const AzBytArr &s_nm, T v) {
    name_value(s, ss, s_nm.c_str(), v); 
  }
  template <class T0, class T1> static void name_values(AzBytArr &s, const AzBytArr &ss, 
                                  const AzBytArr &s_nm, T0 v0, T1 v1) {
    name_value(s, ss, s_nm, v0);                                     
    s << "," << v1; 
  }  
};

template <class Ev>
class AzpG_ClasEvalArray : public virtual AzpG_ClasEval__ {
protected: 
  AzDataArr<Ev> arr; 

public:  
/*------------------------------------------------------------*/
#define kw_clas_fn "classif_fn="
#define kw_real_y_fn "real_label_fn="
/*------------------------------------------------------------*/
  virtual int size() const { return arr.size(); }
  virtual void reset(const AzOut &out, AzParam &azp, const AzpReNet *clas_net_tmpl) {
    AzBytArr s_clas_fn, s_real_y_fn; 
    {
      AzPrint o(log_out); 
      azp.vStr(o, kw_clas_fn, s_clas_fn); 
      if (s_clas_fn.length() <= 0) return; 
      azp.vStr_prt_if_not_empty(o, kw_real_y_fn, s_real_y_fn);  
    }
  
    AzStrPool sp_clas_fn, sp_y_fn; 
    AzTools::getStrings(s_clas_fn.c_str(), '+', &sp_clas_fn); 
    AzTools::getStrings(s_real_y_fn.c_str(), '+', &sp_y_fn);     
    
    arr.reset(sp_clas_fn.size());  
    AzX::throw_if(sp_y_fn.size() > arr.size(), "AzpG_ClasEvalArray::reset", "too many y files?!"); 
    for (int ix = 0; ix < sp_clas_fn.size(); ++ix) {
      const char *clas_fn = sp_clas_fn.c_str(ix);   
      arr(ix)->reset_clas_net(out, clas_fn, clas_net_tmpl); 
      if (ix < sp_y_fn.size()) {
        arr(ix)->reset_y_fn(sp_y_fn.c_str(ix)); 
      }
      arr(ix)->resetParam(ix, out, azp); 
    }
  }      
  void init(const AzOut &out, const AzpData_ *r_trn, const AzpData_ *r_tst, int test_minib) {
    for (int ix = 0; ix < arr.size(); ++ix) arr(ix)->init(ix, out, r_trn, r_tst, test_minib);
  }
  void eval_init(int num) {
    for (int ix = 0; ix < arr.size(); ++ix) arr(ix)->eval_init(num); 
  }
  void eval_clas(const AzPmatVar &mv_gen, int dx_begin, int d_num) {
    for (int ix = 0; ix < arr.size(); ++ix) arr(ix)->eval_clas(mv_gen, dx_begin, d_num);     
  }
  void show_eval(AzBytArr &s, const char *pfx="") {
    for (int ix = 0; ix < arr.size(); ++ix) arr(ix)->show_eval(s, pfx); 
  }
  void resetParam(const AzOut &out, AzParam &azp) {
    for (int ix = 0; ix < arr.size(); ++ix) arr(ix)->resetParam(out, azp);  
  }
}; 

typedef AzpG_ClasEvalArray<AzpG_ClasEval> AzpG_ClasEvalArr; 

#endif 