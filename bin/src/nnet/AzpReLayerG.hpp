/* * * * *
 *  AzpReLayerG.hpp
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
#ifndef _AZP_RE_LAYER_G_HPP_
#define _AZP_RE_LAYER_G_HPP_

#include "AzpReLayer.hpp"
#include "AzpBchNorm.hpp"

/*------------------------------------------------------------*/  
#define AzpReLayer_Type_Patch2D "Patch2D"
#define AzpReLayer_Type_Pooling2D "Pooling2D"
#define AzpReLayer_Type_Reshape "Reshape"
#define AzpReLayer_Type_BchNorm "BchNorm"
/*------------------------------------------------------------*/ 
/*------------------------------------------------------------*/ 
class for_AzpReLayer {
protected: 
  #define kw_xsz "size_x="
  #define kw_ysz "size_y="
  static void resetParam_2D(AzParam &azp, const char *pfx, int &xsz, int &ysz) {
    azp.reset_prefix(pfx); 
    azp.vInt(kw_xsz, &xsz); 
    azp.vInt(kw_ysz, &ysz); 
    azp.reset_prefix(); 
  }
  static void printParam_2D(const AzOut &out, const char *pfx, int xsz, int ysz) {
    AzPrint o(out); 
    o.reset_prefix(pfx);    
    o.printV(kw_xsz, xsz); 
    o.printV(kw_ysz, ysz); 
    o.ppEnd();     
  }
public:  
  static void getSize_2D(const AzOut &out, AzParam &azp, const char *eyec, 
                const AzPfx &pfx, int &xsz, int &ysz) {
    for (int px=0; px<pfx.size(); ++px) resetParam_2D(azp, pfx[px], xsz, ysz); 
    AzXi::throw_if_nonpositive(xsz, eyec, kw_xsz, pfx.pfx()); 
    if (ysz <= 0) ysz = xsz; 
    AzXi::throw_if_nonpositive(ysz, eyec, kw_ysz, pfx.pfx());     
    printParam_2D(out, pfx.pfx(), xsz, ysz);   
  }
  static void to_PmatVar(const char *eyec, int data_num, const AzPmat &m, AzPmatVar &mv) {
    int sz = m.colNum() / data_num; 
    AzX::throw_if(m.colNum() % data_num != 0, eyec, "to_PmatVar detected error."); 
    AzIntArr ia_ind; 
    int col = 0; 
    for (int dx=0; dx<data_num; ++dx, col+=sz) { ia_ind.put(col); ia_ind.put(col+sz); }
    mv.set(&m, &ia_ind);   
  }  
};   
  
/*------------------------------------------------------------*/ 
class AzpReLayer_Patch2D : public virtual AzpReLayer_NoWei_ {
protected:  
  for_AzpReLayer util; 
  int input_sz; 
  void resetParam(const AzOut &out, AzParam &azp, const AzPfx &pfx, int &xsz, int &ysz) {
    util.getSize_2D(out, azp, "AzpReLayer_Patch2D::resetParam", pfx, xsz, ysz);     
    cs.patch->resetParam(azp, pfx); 
    cs.patch->printParam(out, pfx);     
  }
  void show_inout() {
    cs.patch->show_input("   input: ", out); cs.patch->show_output("   output: ", out); 
    AzBytArr s("   "); s << cs.patch->get_channels() << " -> " << cs.patch->patch_length(); 
    AzPrint::writeln(out,   s);     
  }
  
public:
  AzpReLayer_Patch2D() : input_sz(0) { s_nm.reset(AzpReLayer_Type_Patch2D); }
  virtual ~AzpReLayer_Patch2D() {}  
  virtual int nowei_setup(AzParam &azp, const AzpReLayer_Param &pp, const AzPfx &pfx, 
                     bool is_warmstart, bool for_testonly) {
    if (is_warmstart) {
      cs.patch->printParam(out, pfx); 
    }    
    else {
      AzX::no_support(pp.is_spa, "AzpReLayer_Patch2D::nowei_setup", "Sparse input to a Patch2D layer");     
      int xsz = 0, ysz = 0; 
      resetParam(out, azp, pfx, xsz, ysz); 
      AzxD input(xsz, ysz); 
      cs.patch->reset(out, &input, pp.cc, pp.is_spa, false);   
    }
    input_sz = cs.patch->input_region()->size(); 
    show_inout();     
    return cs.patch->patch_length(); 
  }                                
  virtual void upward(bool is_test, const AzDataArr<AzpDataVar_X> &data, AzPmatVar &mv_out, const AzPmatVar *mv2=NULL) {
    const char *eyec = "AzpReLayer_Patch2D::upward(Data_X)";  
    const AzpDataVar_X *dt = data[0]; 
    AzX::no_support(dt->is_spa(), eyec, "Sparse input"); 
    upward(is_test, *dt->den(), mv_out, mv2); 
  }  
  virtual void upward(bool is_test, const AzPmatVar &mv_inp, AzPmatVar &mv_out, const AzPmatVar *mv2) {
    const char *eyec = "AzpReLayer_Patch2D::upward"; 
    no_support2(mv2, eyec);   
    for (int dx = 0; dx < mv_inp.dataNum(); ++dx) {
      if (mv_inp.size(dx) != input_sz) {
        AzBytArr s("Unexpected input size in layer#"); 
        s << layer_no << ", expected " << input_sz << " and got " << mv_inp.size(dx); 
        AzX::throw_if(true, eyec, s.c_str()); 
      }
    }
    AzPmat m; cs.patch->upward(is_test, mv_inp.data(), &m); 
    util.to_PmatVar(eyec, mv_inp.dataNum(), m, mv_out);     
  }
  virtual void get_ld(int id, AzPmatVar &mv_ld, bool do_x2=false) const {
    AzPmatVar mv; upper->get_ld(layer_no, mv); 
    AzPmat m; cs.patch->downward(mv.data(), &m); 
    util.to_PmatVar("AzpReLayer_Patch2D::get_ld", mv.dataNum(), m, mv_ld); 
  } 
};

/*------------------------------------------------------------*/ 
class AzpReLayer_Pooling2D : public virtual AzpReLayer_NoWei_ {
protected:  
  int input_sz; 
  for_AzpReLayer util; 

  void resetParam(const AzOut &out, AzParam &azp, const AzPfx &pfx, int &xsz, int &ysz) {
    util.getSize_2D(out, azp, "AzpReLayer_Patch2D::resetParam", pfx, xsz, ysz);     
    cs.pool->resetParam(azp, pfx); 
    cs.pool->printParam(out, pfx);     
  }
public:
  AzpReLayer_Pooling2D() : input_sz(0) { s_nm.reset(AzpReLayer_Type_Pooling2D); }
  virtual ~AzpReLayer_Pooling2D() {}  
  virtual int nowei_setup(AzParam &azp, const AzpReLayer_Param &pp, const AzPfx &pfx, 
                     bool is_warmstart, bool for_testonly) {
    AzX::no_support(pp.is_spa, "AzpReLayer_Pooling2D::nowei_setup", "Sparse input to a Pooling2D layer");     
    if (is_warmstart) {
      cs.pool->printParam(out, pfx); 
    }
    else {
      int xsz = 0, ysz = 0; 
      resetParam(out, azp, pfx, xsz, ysz); 
      AzxD input(xsz, ysz); 
      cs.pool->reset(out, &input, NULL);   
    }
    input_sz = cs.pool->input_size(); 
    return pp.cc; 
  }                
  virtual bool can_do_up2() const { return cs.pool->can_do_up2(); }
  virtual void upward(bool is_test, const AzPmatVar &mv_inp, AzPmatVar &mv_out, const AzPmatVar *mv2) {  
    no_support2(mv2, "AzpReLayer_Pooling2D::upward"); 
    _up(is_test, mv_inp, mv_out); 
  }
  virtual void upward2(const AzPmatVar &mv_inp, AzPmatVar &mv_out, bool dont_update=false) {
    AzX::no_support(!can_do_up2(), "AzpReLayer_Pooling2D::upward2", "upward2"); 
    bool do_up2 = true; 
    _up(true, mv_inp, mv_out, do_up2); 
  }
protected:  
  virtual void _up(bool is_test, const AzPmatVar &mv_inp, AzPmatVar &mv_out, bool do_up2=false) {
    const char *eyec = "AzpReLayer_Pooling2D::_up";   
    for (int dx = 0; dx < mv_inp.dataNum(); ++dx) {
      if (mv_inp.size(dx) != input_sz) {
        AzBytArr s("Unexpected input size in layer#"); 
        s << layer_no << ", expected " << input_sz << " and got " << mv_inp.size(dx); 
        AzX::throw_if(true, eyec, s.c_str()); 
      }
    }
    AzPmat m; 
    if (do_up2) cs.pool->upward2(*mv_inp.data(), m);  
    else        cs.pool->upward(is_test, mv_inp.data(), &m); 
    util.to_PmatVar(eyec, mv_inp.dataNum(), m, mv_out);     
  }
public:  
  virtual void get_ld(int id, AzPmatVar &mv_ld, bool do_x2=false) const {
    AzPmatVar mv; upper->get_ld(layer_no, mv); 
    AzPmat m; cs.pool->downward(mv.data(), &m); 
    util.to_PmatVar("AzpReLayer_Pooling2D::get_ld", mv.dataNum(), m, mv_ld); 
  } 
};

/*------------------------------------------------------------*/ 
class AzpReLayer_Reshape : public virtual AzpReLayer_NoWei_ {
protected:  
  int out_rnum, dsno; /* parameters */
  int inp_rnum; 
  
  #define kw_out_rnum "num_rows="
  void resetParam(AzParam &azp, const char *pfx) {
    azp.reset_prefix(pfx); 
    azp.vInt(kw_dsno, &dsno); 
    azp.vInt(kw_out_rnum, &out_rnum); 
    azp.reset_prefix();     
  }
  void resetParam(const AzOut &out, AzParam &azp, const AzPfx &pfx) {
    for (int px=0; px<pfx.size(); ++px) resetParam(azp, pfx[px]); 
    AzXi::throw_if_nonpositive(out_rnum, "AzpReLayer_Reshape::resetParam", kw_out_rnum);     
    printParam(out, pfx.pfx()); 
  }
  void printParam(const AzOut &out, const char *pfx) const {
    AzPrint o(out, pfx); 
    o.printV(kw_dsno, dsno); 
    o.printV(kw_out_rnum, out_rnum); 
    o.ppEnd();     
  }
public:
  AzpReLayer_Reshape() : out_rnum(-1), inp_rnum(-1), dsno(-1) { s_nm.reset(AzpReLayer_Type_Reshape); }
  virtual ~AzpReLayer_Reshape() {}  
  virtual void _write(AzFile *file) const { 
    file->writeInt(layer_no); 
    file->writeInt(out_rnum); 
    file->writeInt(dsno); 
  }
  virtual void _read(AzFile *file) { 
    layer_no = file->readInt(); 
    out_rnum = file->readInt(); 
    dsno = file->readInt(); 
  }  
  virtual int nowei_setup(AzParam &azp, const AzpReLayer_Param &pp, const AzPfx &pfx, 
                     bool is_warmstart, bool for_testonly) {
    if (!is_warmstart) {                      
      AzX::no_support(pp.is_spa, "AzpReLayer_Reshape::_nowei_setup", "Sparse input to a Shape layer");     
      resetParam(out, azp, pfx); 
    }
    return out_rnum; 
  }                
  virtual void upward(bool is_test, const AzDataArr<AzpDataVar_X> &arr, AzPmatVar &mvo, const AzPmatVar *mv2=NULL) {
    AzX::throw_if(dsno >= arr.size(), "AzpReLayer_Reshape::upward(data)", "data# is out of range"); 
    AzPmatVar mv; arr[MAX(0,dsno)]->get(mv); 
    upward(is_test, mv, mvo, mv2); 
  }  
  virtual void upward(bool is_test, const AzPmatVar &mv_inp, AzPmatVar &mv_out, const AzPmatVar *mv2) {
    const char *eyec = "AzpReLayer_Reshape::upward"; 
    no_support2(mv2, eyec);   
    inp_rnum = mv_inp.rowNum(); 
    _change_dim(mv_inp, out_rnum, mv_out, eyec); 
  }
  virtual void get_ld(int id, AzPmatVar &mv_ld, bool do_x2=false) const {
    AzPmatVar mv; upper->get_ld(layer_no, mv); 
    _change_dim(mv, inp_rnum, mv_ld, "AzpReLayer_Reshape::get_ld"); 
  } 
protected: 
  static void _change_dim(const AzPmatVar &mv_inp, int onum, AzPmatVar &mv_out, const char *eyec) {
    int inum = mv_inp.rowNum(); 
    AzX::throw_if(mv_inp.data()->size()%onum != 0, eyec, "Wrong shape"); 
    AzPmat m(mv_inp.data()); 
    m.change_dim(onum, mv_inp.data()->size()/onum); 
    AzIntArr ia_ind; 
    int col = 0; 
    for (int dx = 0; dx < mv_inp.dataNum(); ++dx) {
      ia_ind.put(col); 
      int sz = mv_inp.size(dx) * inum; 
      AzX::throw_if(sz%onum != 0, eyec, "Wrong data shape"); 
      col += sz/onum; 
      ia_ind.put(col); 
    }   
    mv_out.set(&m, &ia_ind);   
  }
};

/*------------------------------------------------------------*/ 
#if 0 
class AzpReLayer_BchNorm : public virtual AzpReLayer_NoWei_ {
#else  
template <class Cls> 
class AzpReLayer_Norm : public virtual AzpReLayer_NoWei_ {  
#endif 
protected:  
#if 0 
  AzpBchNorm bn;
#else
  Cls bn; 
#endif  
  AzPmatVar mv_ld;   
public:
#if 0 
  AzpReLayer_BchNorm() { s_nm.reset(AzpReLayer_Type_BchNorm); }  
  virtual ~AzpReLayer_BchNorm() {} 
#else  
  AzpReLayer_Norm() { s_nm.reset(bn.name()); }  
  virtual ~AzpReLayer_Norm() {} 
#endif   
  virtual void _write(AzFile *file) const { file->writeInt(layer_no); bn.write(file); }
  virtual void _read(AzFile *file) { layer_no = file->readInt(); bn.read(cs, file); }
  virtual void upward(bool is_test, const AzPmatVar &mv_below, AzPmatVar &mv_out, const AzPmatVar *mv2=NULL) {
    mv_out.set(&mv_below); 
    bn.upward(is_test, *mv_out.data_u()); 
  }
  virtual void upward2(const AzPmatVar &mv_inp, AzPmatVar &mv_out, bool dont_update=false) {
    mv_out.set(&mv_inp); 
    bn.upward2(*mv_inp.data(), *mv_out.data_u(), mv_inp.dataNum(), dont_update);  
  }
  virtual void get_ld(int id, AzPmatVar &mv_lossd_a, bool do_x2=false) const { mv_lossd_a.set(&mv_ld); }
  virtual void downward(const AzPmatVar &mv_loss_deriv, bool dont_update=false, bool dont_release_sv=false) {
    upper->get_ld(layer_no, mv_ld); 
    bn.downward(*mv_ld.data_u(), mv_loss_deriv.dataNum(), dont_update, dont_release_sv);  
  }  
  virtual void flushDelta() { bn.flushDelta(); }  
  virtual void show_stat(AzBytArr &s) const { bn.show_stat(s); }
  virtual void multiply_to_stepsize(double coeff, const AzOut *out) { bn.multiply_to_stepsize(coeff); }
  virtual void release_sv() { bn.release_sv(); }
  virtual void release_ld() { bn.release_ld(); mv_ld.destroy(); }
protected:   
  virtual int nowei_setup(AzParam &azp, const AzpReLayer_Param &pp, const AzPfx &pfx, 
                     bool is_warmstart, bool for_testonly) {
    bn.reset(cs, pp.cc, out, azp, pfx, is_warmstart, for_testonly);  
    return pp.cc; 
  }
};
typedef AzpReLayer_Norm<AzpBchNorm> AzpReLayer_BchNorm; 

/*------------------------------------------------------------*/  
class AzpReLayersG : public virtual AzpReLayers {
protected: 
  AzBytArr s_eyec; 
public: 
  AzpReLayersG() : s_eyec("AzpReLayersG") {}
  virtual ~AzpReLayersG() { reset(); }

protected: 
  void _init_layer(int index, const AzpCompoSet_ *cset) {
    const char *eyec = "_init_layer";
    check_index(index, eyec);
    AzBytArr s_typ(as_typ[index]);
    if (s_typ.length() <= 0) s_typ.reset(AzpReLayer_Type_Fc);
    AzpReLayer_ *ptr = NULL;
    if      (s_typ.equals(AzpReLayer_Type_BchNorm))   ptr = new AzpReLayer_BchNorm();
    else if (s_typ.equals(AzpReLayer_Type_Patch2D))   ptr = new AzpReLayer_Patch2D();
    else if (s_typ.equals(AzpReLayer_Type_Pooling2D)) ptr = new AzpReLayer_Pooling2D();
    else if (s_typ.equals(AzpReLayer_Type_Reshape))   ptr = new AzpReLayer_Reshape();
    else if (s_typ.equals("BNorm"))                   ptr = new AzpReLayer_BchNorm(); /* for compatibility */    
    else if (s_typ.equals("DWei"))                    ptr = new AzpReLayer_DenWei();  /* for compatibility */
    if (ptr == NULL) {
      AzpReLayers::_init_layer(index, cset); 
    }
    else {   
      ptr->reset(cset); 
      a.set(index, ptr); 
    }
  }  
};
#endif 
