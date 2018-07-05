/* * * * *
 *  AzpData_tmpl_.hpp
 *  Copyright (C) 2017,2018 Rie Johnson
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

#ifndef _AZP_DATA_TMPL_HPP_
#define _AZP_DATA_TMPL_HPP_

#include "AzUtil.hpp"
#include "AzDic.hpp"
                      
class AzpData_tmpl_ {
public: 
  virtual int datasetNum() const = 0; 
  virtual int xdim(int dsno=0) const  = 0; 
  virtual const AzDicc &get_dic(int dsno=0) const = 0;  
  virtual bool is_vg_x() const = 0; 
  virtual bool is_sparse_x() const = 0;  
  virtual int ydim() const = 0; 
  virtual int dimensionality() const = 0; 
  virtual int size(int index) const = 0; 
  virtual void signature(AzBytArr &s) const {
    s.reset(); 
    s << "dim:" << dimensionality() << ";"; 
    s << "channel:" << xdim() << ";"; 
    for (int ix = 0; ix < dimensionality(); ++ix) {
      s << "size" << ix << ":" << size(ix) << ";"; 
    }
  }
  virtual bool isSignatureCompatible(const AzBytArr &s_nn, const AzBytArr &s_data) const {
    if (s_nn.compare(&s_data) == 0) return true; 
    AzStrPool sp1(10,10), sp2(10,10); 
    AzTools::getStrings(s_nn.c_str(), ';', &sp1); 
    AzTools::getStrings(s_data.c_str(), ';', &sp2); 
    if (sp1.size() != sp2.size()) {
      return false; 
    }
    for (int ix = 0; ix < sp1.size(); ++ix) {
      if (strcmp(sp1.c_str(ix), sp2.c_str(ix)) != 0) {
        AzBytArr s1; sp1.get(ix, &s1); 
        AzBytArr s2; sp2.get(ix, &s2); 
        if (ix == 0 && s1.length() != s2.length()) {
          /*---  sparse_multi with one dataset is equivalent to sparse  ---*/
          if (s1.beginsWith("[0]") && s2.compare(s1.point()+3, s1.length()-3) == 0) continue; 
          if (s2.beginsWith("[0]") && s1.compare(s2.point()+3, s2.length()-3) == 0) continue; 
        }
        else if (s1.beginsWith("size") && s2.beginsWith("size") && s1.endsWith("-1")) {
          /*---  nn is trained for variable-size input, and data is fixed-size ---*/
          continue; 
        }
        else return false; 
      }
    }
    return true; 
  }  
  virtual void get_info(AzxD &data_dim) const {
    AzIntArr ia_sz; 
    int dim = dimensionality(); 
    for (int ix = 0; ix < dim; ++ix) ia_sz.put(size(ix)); 
    data_dim.reset(&ia_sz); 
  }
  virtual bool is_same_x_tmpl(const AzpData_tmpl_ &i) const {
    if (datasetNum() != i.datasetNum()) return false; 
    for (int ix=0; ix<datasetNum(); ++ix) {
      if (xdim(ix) != i.xdim(ix)) return false;     
      if (!get_dic(ix).is_same(i.get_dic(ix))) return false; 
    }
    if (is_vg_x() != i.is_vg_x()) return false; 
    if (is_sparse_x() != i.is_sparse_x()) return false; 
    if (dimensionality() != i.dimensionality()) return false; 
    for (int ix=0; ix<dimensionality(); ++ix) if (size(ix) != i.size(ix)) return false; 
    return true; 
  }  
  virtual bool is_same_y_tmpl(const AzpData_tmpl_ &i) const {
    if (ydim() != i.ydim()) return false; 
    return true; 
  }  
};  

/*---  to save a data template to a file  ---*/
class AzpData_tmpl_save : public virtual AzpData_tmpl_ {
protected: 
  int _datasetNum, _xdim, _ydim, _dimensionality; 
  bool _is_vg_x, _is_sparse_x; 
  AzIntArr ia_size; 
  AzDicc _dic;
  
  static const int version = 0;   
  static const int reserved_len=64;  
public: 
  AzpData_tmpl_save() : _datasetNum(-1), _xdim(-1), _ydim(-1), _dimensionality(-1), 
                    _is_vg_x(false), _is_sparse_x(false) {}
  virtual int datasetNum() const { return _datasetNum; }
  virtual int xdim(int dsno=0) const  { return _xdim; }
  virtual const AzDicc &get_dic(int dsno=0) const { return _dic; } 
  virtual bool is_vg_x() const { return _is_vg_x; }
  virtual bool is_sparse_x() const { return _is_sparse_x; }
  virtual int ydim() const { return _ydim; }
  virtual int dimensionality() const { return _dimensionality; }
  virtual int size(int index) const { 
    AzX::throw_if(index<0 || index>=ia_size.size(), "AzpData_tmpl_save::size", "index is out of range"); 
    return ia_size[index]; 
  } 
  
  /*---  ---*/
  bool is_set() const { return (_dimensionality > 0); }
  void reset(const AzpData_tmpl_ &i) {
    _datasetNum = i.datasetNum(); 
    _xdim = i.xdim(); _ydim = i.ydim(); 
    _dimensionality = i.dimensionality(); 
    _is_vg_x = i.is_vg_x(); 
    _is_sparse_x = i.is_sparse_x(); 
    ia_size.reset(); 
    for (int ix = 0; ix < _dimensionality; ++ix) ia_size.put(i.size(ix)); 
    _dic.reset(i.get_dic());     
  }
  
  void write(AzFile *file) const {
    AzX::throw_if(!is_set(), "AzpData_tmpl_save::write", "Not set"); 
    AzTools::write_header(file, version, reserved_len);     
    file->writeInt(_datasetNum); 
    file->writeInt(_xdim); 
    file->writeInt(_ydim); 
    file->writeInt(_dimensionality); 
    file->writeBool(_is_vg_x); 
    file->writeBool(_is_sparse_x); 
    ia_size.write(file); 
    _dic.write(file); 
  }
  void read(AzFile *file) {
    AzTools::read_header(file, reserved_len); 
    _datasetNum = file->readInt(); 
    _xdim = file->readInt(); 
    _ydim = file->readInt(); 
    _dimensionality = file->readInt();     
    _is_vg_x = file->readBool(); 
    _is_sparse_x = file->readBool(); 
    ia_size.read(file); 
    _dic.read(file); 
  }
}; 

class AzpData_tmpl1D : public virtual AzpData_tmpl_ {
protected: 
  AzDicc dummy_dic; 
  int _xdim;
public:
  AzpData_tmpl1D(int ixdim) : _xdim(1) { _xdim = ixdim; }   
  virtual void reset_xdim(int ixdim) { _xdim = ixdim; }  
  virtual int xdim(int dsno=0) const { return _xdim; }  
  virtual bool is_vg_x() const { return true; }
  virtual bool is_sparse_x() const { return false; }  
  virtual int ydim() const { return 1; }
  virtual int datasetNum() const { return 1; }
  virtual int dimensionality() const { return 1; }
  virtual int size(int index) const { return -1; }
  virtual const AzDicc &get_dic(int dsno=0) const { return dummy_dic; }
}; 

class AzpData_tmpl1D2 : public virtual AzpData_tmpl_ {
protected: 
  AzDicc dummy_dic; 
  int _xdim0, _xdim1; 
public:
  AzpData_tmpl1D2(int ixdim0, int ixdim1) { reset_xdim(ixdim0, ixdim1); }
  void reset_xdim(int ixdim0, int ixdim1) { _xdim0=ixdim0; _xdim1=ixdim1; }
  virtual int xdim(int dsno=0) const { 
    if (dsno == 0) return _xdim0; 
    else           return _xdim1; 
  }  
  virtual bool is_vg_x() const { return true; }
  virtual bool is_sparse_x() const { return false; }  
  virtual int ydim() const { return 1; }
  virtual int datasetNum() const { return 2; }
  virtual int dimensionality() const { return 1; }
  virtual int size(int index) const { return -1; }
  virtual const AzDicc &get_dic(int dsno=0) const { return dummy_dic; }
}; 
#endif 