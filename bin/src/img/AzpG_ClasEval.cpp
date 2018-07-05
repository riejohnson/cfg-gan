/* * * * *
 *  AzpG_ClasEval.cpp
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
#include "AzpG_ClasEval.hpp"

/*------------------------------------------------------------*/ 
void AzpG_ClasEval::eval_clas(const AzPmatVar &mv_gen, int dx_begin, int d_num) {
  if (clas_net == NULL) return;                                

  const char *eyec = "AzpG_ClasEval::eval_clas"; 
  bool is_test = true;                                
  AzPmatVar mv_out; 
  clas_net->up(is_test, mv_gen, mv_out); 
    
  AzX::throw_if(mv_out.colNum() != d_num, eyec, "#output mismatch"); 
  AzX::throw_if(m_out.rowNum() != mv_out.rowNum() || m_out.colNum() < dx_begin+d_num, 
                eyec, "dim mismatch (m_out)"); 
  m_out.set(dx_begin, d_num, mv_out.data());      
}

/*------------------------------------------------------------*/ 
double AzpG_ClasEval::_eval_score(const AzPmat &_m, AzPmat *_v_sclas) { /* static */
  AzPmat m_out(&_m); 
  double eps = 1e-20; 
  m_out.exp(); m_out.add(eps); m_out.normalize1(); _smooth(m_out, eps); /* prob */
  AzPmat v_py; v_py.rowSum(&m_out); v_py.normalize1(); _smooth(v_py, eps); /* p(y) */
  if (_v_sclas != NULL) _v_sclas->set(&v_py); 
 
  v_py.pow(-1); /* 1/p(y) */
  AzPmat m(&m_out); m.multi_col(&v_py); m.log(); /* log(p(y|x_i)/p(y)) */
  m.elm_multi(&m_out); /* p(y|x_i) log(p(y|x_i)/p(y)) */
  double val = exp(m.sum() / (double)m.colNum()); 
  return val; 
}

/*------------------------------------------------------------*/ 
void AzpG_ClasEval::_smooth(AzPmat &m, double eps) { /* static */
  if (m.size() <= 0) return; 
  while(m.min() == 0) {
    m.add(eps); 
    m.normalize1();     
  }
}

/*------------------------------------------------------------*/ 
double AzpG_ClasEval::_get_kl(const AzPmat &_v0, const AzPmat &_v1) const {
  double eps = 1e-20; 
  AzPmat v1(&_v1); v1.add(eps); v1.normalize1(); _smooth(v1, eps); 
  AzPmat v0(&_v0); v0.add(eps); v0.normalize1(); _smooth(v0, eps); 
  bool do_inv = true; 
  AzPmat v(&v0);  v.elm_multi(&v1, do_inv); 
  v.log(); 
  v.elm_multi(&v0); 
  return v.sum(); 
}   

/*------------------------------------------------------------*/ 
void AzpG_ClasEval::_init_real_y(const AzOut &out, const char *y_fn) {
  if (clas_net == NULL) return; 
  AzPrint::writeln(out, ""); 
  AzPrint::writeln(out, "init_real_y, y_fn=", y_fn); 
  
  AzSmat m_y; 
  AzTextMat::readMatrix(y_fn, &m_y); 
  AzX::throw_if(clas_net->classNum() != m_y.rowNum(), AzInputError, 
                "AzpG_ClasEval::_init_real_y(file)", 
                "#class conflict btw clas_net and y file", y_fn); 
  AzDvect v_y(m_y.rowNum()); 
  for (int col = 0; col < m_y.colNum(); ++col) v_y.add(m_y.col(col)); 
  v_real_y.set(&v_y); 
  v_real_y.dump(out, "real_y"); 
}
 
/*------------------------------------------------------------*/ 
void AzpG_ClasEval::show_eval(AzBytArr &s, const char *pfx) {
  AzBytArr ss(pfx); 
  if (myno > 0) ss << myno; 

  _show_scores(s, ss); 
}   

/*------------------------------------------------------------*/ 
void AzpG_ClasEval::_show_scores(AzBytArr &s, const AzBytArr &ss) {
  AzPmat v_sclas; /* s for soft */
  double score = _eval_score(m_out, &v_sclas);  
  name_value(s, ss, s_snm, score); 
  
  if (v_real_y.size() > 0) {
    AzPmat vv_clas(&v_sclas); 
    double kl0 = _get_kl(v_real_y, vv_clas); 
    double kl1 = _get_kl(vv_clas, v_real_y); 
    name_values(s, ss, s_cnm, kl0, kl1); 
  }
} 

/*------------------------------------------------------------*/ 
/* static */
void AzpG_ClasEval_::eval_img(const AzOut &out, bool do_verbose, 
                             AzpG_ClasEval_ &cevs, const AzpData_ *tst, 
                             int mb, const AzIntArr *_ia_dxs, const AzpData_ *data_for_init) {
  const char *eyec = "AzpG_ClasEval_::eval_img"; 
  AzX::throw_if_null(tst, eyec, "test data"); 
  AzX::no_support(tst->batchNum() > 1, eyec, "multiple data batches");   
  AzBytArr s("image size: "); s << tst->size(0) << " x " << tst->size(1) << " x " << tst->xdim();
  AzTimeLog::print(s.c_str(), out); 
 
  const AzIntArr *ia_dxs = _ia_dxs; AzIntArr ia; 
  if (ia_dxs == NULL) { ia.range(0, tst->dataNum()); ia_dxs = &ia; }
  int data_num = ia_dxs->size(); 
  bool do_ml = (ia_dxs->size() >= 10000); 
  
  AzOut less_out; if (do_verbose) less_out = out; 
  if (data_for_init != NULL) cevs.init(less_out, tst, data_for_init, mb); 
  else                       cevs.init(less_out, tst, tst, mb); 
  cevs.eval_init(data_num);
  AzPmat m_all_pred; 
  int mlinc = data_num / 50, milestone = mlinc; 
  AzTimeLog::print("Evaluating ... ", out);   
  for (int ix = 0; ix < data_num; ix += mb) {
    if (do_ml) AzTools::check_milestone(out, milestone, ix, mlinc);     
    int d_num = MIN(mb, data_num-ix); 
    AzPmatVar mv_img; 
    tst->gen_data(ia_dxs->point()+ix, d_num, mv_img); 
    cevs.eval_clas(mv_img, ix, d_num);
  }    
  if (do_ml) AzTools::finish_milestone(out, milestone); 
    
  s.reset(); 
  s << "#images," << data_num; 
  cevs.show_eval(s);
  AzPrint::writeln(out, s); 
}
