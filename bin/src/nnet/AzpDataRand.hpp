/* * * * *
 *  AzpDataRand.hpp
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

#ifndef _AZP_DATA_RAND_HPP_
#define _AZP_DATA_RAND_HPP_

#include "AzRandGen.hpp"
#include "AzpData_tmpl_.hpp"
#include "AzPmat.hpp"

/* generate random vectors for input to generative models */
class AzpDataRand : public virtual AzpData_tmpl_ {
protected: 
  AzRandGen rg; 
  int rnum, data_seed; 
  double normal;   /* mean 0, stddev x */
  double uniform;  /* [-x,x] */
  double uniform0; /* [0,x] */
  AzDicc dummy_dic; 
  static const int version = 0;   
  static const int reserved_len=64;     
public:
  AzpDataRand() : rnum(0), normal(-1), uniform(-1), uniform0(-1), data_seed(-1) {
    rg._srand_(data_seed); 
  }
  virtual void reset(const AzOut &out, AzParam &azp, const char *pfx="", int seed=-1) {
    resetParam(azp, pfx); 
    data_seed = seed; 
    printParam(out, pfx); 
    init(); 
  }        
  virtual void init() {
    rg._srand_(data_seed); 
  }  
  virtual void reset_seed(int seed) { data_seed = seed; init(); }
  virtual int get_seed() const { return data_seed; }  
  /*------------------------------------------*/   
  #define kw_rnum "num_rows="
  #define kw_uniform "uniform="
  #define kw_uniform0 "uniform0="
  #define kw_normal "normal="
  /*------------------------------------------------*/
  virtual void resetParam(AzParam &azp, const char *pfx="") {
    const char *eyec = "AzpDataRand::resetParam"; 
    azp.reset_prefix(pfx); 
    azp.vInt(kw_rnum, &rnum); 
    azp.vFloat(kw_normal, &normal); 
    if (normal <= 0) {
      azp.vFloat(kw_uniform, &uniform);    
      if (uniform <= 0) azp.vFloat(kw_uniform0, &uniform0); 
    }
    azp.reset_prefix();       
    AzX::throw_if(normal <= 0 && uniform <= 0 && uniform0 <= 0, eyec, "Either uniform, uniform0, or normal should be positive."); 
    AzXi::throw_if_nonpositive(rnum, eyec, kw_rnum);      
  }
  virtual void printParam(const AzOut &out, const char *pfx="") const {
    if (out.isNull()) return; 
    AzPrint o(out, pfx); 
    o.printV(kw_rnum, rnum); 
    o.printV(kw_normal, normal); 
    o.printV(kw_uniform, uniform); 
    o.printV(kw_uniform0, uniform0);     
  }   
  virtual void gen_data(int d_num, AzDataArr<AzpDataVar_X> &darr) const {  
    darr.reset(datasetNum()); 
    for (int dsno = 0; dsno < darr.size(); ++dsno) {
      gen_data(d_num, darr(dsno), dsno); 
    }
  }
  virtual void gen_data(int d_num, AzpDataVar_X *data, int dsno=0) const {  
    AzX::throw_if_null(data, "AzpDataRand::gen_data", "data"); 
    AzPmat m; gen_data(d_num, m, dsno); 
    AzpData_::set_var(m, *data); 
  }
  virtual void gen_data(int d_num, AzPmatVar &mv_data, int dsno=0) const {
    AzPmat m; gen_data(d_num, m, dsno); 
    mv_data.set(&m); 
  }
  virtual void gen_data(int d_num, AzPmat &m_data, int dsno=0) const {
    AzX::no_support(dsno != 0, "AzpDataRand::gen_data", "dsno<>0");  
    _gen_data0(d_num, m_data);   
  }
protected:   
  virtual void _gen_data0(int d_num, AzPmat &m_data) const {
    m_data.reform(rnum, d_num); 
    if (uniform > 0 || uniform0 > 0) {
      AzDmat m(m_data.rowNum(), m_data.colNum()); 
      (const_cast<AzpDataRand *>(this))->rg.uniform_01(m); 
      m_data.set(&m); 
      if (uniform > 0) { /* [-x,x] */
        m_data.add(-0.5); 
        m_data.multiply(uniform*2); 
      }
      else if (uniform0 != 1) { /* [0,x] */
        m_data.multiply(uniform0); 
      }
    }   
    else {
      AzDmat m(m_data.rowNum(), m_data.colNum()); 
      (const_cast<AzpDataRand *>(this))->rg.gaussian(normal, &m); 
      m_data.set(&m);   
    }
  }   
public:  
  
  /*---  To implement AzpData_tmpl_  ---*/                  
  virtual bool is_vg_x() const { return true; }
  virtual bool is_vg_y() const { return false; }  
  virtual bool is_sparse_x() const { return false; }  
  virtual bool is_sparse_y() const { return true; }
  
  virtual int ydim() const { return 1; }
   
  virtual int datasetNum() const { return 1; }
  virtual int xdim(int dsno=0) const { 
    check_dsno(dsno, "AzpDataRand::channelNum"); 
    return (dsno == 0) ? rnum : -1; 
  }
  virtual int dimensionality() const { return 1; }
  virtual int size(int index) const { return (index == 0) ? 1 : -1; }
  virtual const AzDicc &get_dic(int dsno=0) const { return dummy_dic; }

  virtual void write(AzFile *file) const {
    AzTools::write_header(file, version, reserved_len);      
    file->writeBool(false); /* for compatibility */
    file->writeInt(rnum); 
    file->writeInt(data_seed); 
    file->writeDouble(uniform); 
    file->writeDouble(normal); 
    file->writeBool(false); /* for compatibility */
    file->writeInt(-1); 
  }
  virtual void read(AzFile *file) {
    AzTools::read_header(file, reserved_len); 
    file->readBool(); /* for compatibility */
    rnum = file->readInt(); 
    data_seed = file->readInt(); 
    uniform = file->readDouble(); 
    normal = file->readDouble(); 
    file->readBool();  /* for compatibility */
    file->readInt();    /* for compatibility */ 
  }
  virtual void copy_params_from(const AzpDataRand &i) { /* copy only the things written in a file */
    AzFileV vf("w"); /* open a virtual file */
    i.write(&vf); 
    vf.close(); 
    vf.open("r");    
    read(&vf);    
    vf.close();   
  }
protected:   
  virtual void check_dsno(int dsno, const char *eyec) const {
    AzX::throw_if(dsno < 0 || dsno >= datasetNum(), eyec, "dsno is out of range"); 
  } 
}; 
#endif 
