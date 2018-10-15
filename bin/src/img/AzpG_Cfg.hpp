/* * * * *
 *  AzpG_Cfg.hpp 
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
#ifndef _AZP_G_CFG_HPP_
#define _AZP_G_CFG_HPP_

#include "AzpReNet.hpp"
#include "AzpDataRand.hpp"
#include "AzpG_ClasEval.hpp"

extern AzPdevice dev; 
extern bool __doDebug; 

class AzpG_Ev_ {
public:   
  virtual void add(int typ, const AzPmatVar &mv) = 0; 
  virtual void add(int typ, const AzPmat &m) = 0; 
  virtual void add_val(int typ, double val, int count=1) = 0; 
  virtual void min_val(int typ, double val, int count=1) = 0; 
  virtual void max_val(int typ, double val, int count=1) = 0; 
};

/*----------------------------------------------------------------------*/
#define AzpG_Ev_TypeNum 20
#define AzpG_Ev_Rraw 0
#define AzpG_Ev_Fraw 1
class AzpG_Ev : public virtual AzpG_Ev_ {
protected: 
  AzBytArr ss; 
  double sum[AzpG_Ev_TypeNum], count[AzpG_Ev_TypeNum]; 
public:  
  AzpG_Ev() { reset(); }
  AzpG_Ev(const char *str) { reset(); ss.reset(str); }
  void reset_str(const char *str) { ss.reset(str); }
  void reset() { for (int ix = 0; ix < AzpG_Ev_TypeNum; ++ix) sum[ix]=count[ix]=0; }
  void reset(const AzpG_Ev &i) {
    ss.reset(&i.ss); 
    for (int ix = 0; ix < AzpG_Ev_TypeNum; ++ix) {
      sum[ix] = i.sum[ix]; count[ix] = i.count[ix];  
    }
  }   
  void add(int typ, const AzPmatVar &mv) {  
    check_typ(typ, "add(mv)"); 
    sum[typ] += mv.data()->sum(); count[typ] += mv.dataNum();     
  } 
  void add(int typ, const AzPmat &m) {  
    check_typ(typ, "add(m)"); 
    sum[typ] += m.sum(); count[typ] += m.colNum();     
  }
  virtual void add_val(int typ, double val, int c=1) {
    AzpG_Ev::check_typ(typ, "add_val"); 
    sum[typ]+=val; count[typ]+=c; 
  }
  virtual void max_val(int typ, double val, int c=1) {
    AzpG_Ev::check_typ(typ, "max_val");
    if (count[typ]<=0) { sum[typ]=val;                count[typ]=c; }
    else               { sum[typ]=MAX(sum[typ], val); count[typ]+=c; }
  }
  virtual void min_val(int typ, double val, int c=1) {
    AzpG_Ev::check_typ(typ, "min_val");
    if (count[typ]<=0) { sum[typ]=val;                count[typ]=c; }
    else               { sum[typ]=MIN(sum[typ], val); count[typ]+=c; }
  }
  double avg(int typ) const { 
    check_typ(typ, "avg"); 
    return (count[typ] <= 0) ? 0 : sum[typ]/count[typ]; 
  }
  double get_count(int typ) const { 
    check_typ(typ, "get_count"); 
    return count[typ]; 
  }
  bool is_nan() const {
    for (int ix = 0; ix < AzpG_Ev_TypeNum; ++ix) if (sum[ix] != sum[ix]) return true; 
    return false;     
  }
  virtual void format_eval(AzBytArr &s, bool do_verbose=false) const {   
    double rr=avg(AzpG_Ev_Rraw), rf=avg(AzpG_Ev_Fraw);  
    double rc=count[AzpG_Ev_Rraw], fc=count[AzpG_Ev_Fraw];  
    if (rc > 0 || fc > 0) {
      s << "," <<ss<<"Dreal-Dgen," << rr-rf; 
      s << "," <<ss<<"Dreal," << rr << "," <<ss<<"Dgen," << rf;   
      if (do_verbose) s << "," <<ss<<"c," << rc << "," << fc;       
    }
  }
  
  virtual void show_optional(AzBytArr &s, int typ, const char *str, bool do_verbose) const {
    check_typ(typ, "show_optional"); 
    if (count[typ] > 0) {
      double val = avg(typ); 
      s << "," <<ss<<str << "," << val;; 
      if (do_verbose) s << "," << count[typ]; 
    }
  }  
  virtual void show_optional_minmax(AzBytArr &s, int typ, const char *str, bool do_verbose) const {
    check_typ(typ, "show_optional_minmax"); 
    if (count[typ] > 0) {
      s << "," <<ss<<str << "," << sum[typ]; 
      if (do_verbose) s << "," << count[typ]; 
    }
  }
protected:
  virtual void check_typ(int typ, const char *eyec) const {
    AzX::throw_if(typ<0 || typ>=AzpG_Ev_TypeNum, eyec, "index is out of range");  
  }  
};
 
template <class Ev> 
class AzpG_EvArray : public virtual AzpG_Ev_ {
protected:
  AzDataArr<Ev> evs; 
public:
  AzpG_EvArray() {}
  AzpG_EvArray(int num, const char *str="") { reset(num, str); }
  void reset(int num, const char *str) {
    evs.reset(num); 
    for (int ex = 0; ex < num; ++ex) evs(ex)->reset_str(str);     
  }  
  virtual void format_eval(AzBytArr &s, bool do_verbose, int index=0) {
    check_index(index, "AzpG_EvArray::show"); 
    evs[index]->format_eval(s, do_verbose); 
  }
  int size() const { return evs.size(); }
  const AzpG_Ev *operator[](int index) const {
    check_index(index, "AzpG_EvArray::[]"); 
    return evs[index]; 
  }
  AzpG_Ev *operator()(int index) {
    check_index(index, "AzpG_EvArray::()"); 
    return evs(index); 
  }  
  void add(int typ, const AzPmat &m) { for (int ex=0;ex<evs.size();++ex) evs(ex)->add(typ, m); }  
  void add(int typ, const AzPmatVar &mv) { for (int ex=0;ex<evs.size();++ex) evs(ex)->add(typ, mv); }
  void add_val(int typ,double val,int c=1) { for (int ex=0;ex<evs.size();++ex) evs(ex)->add_val(typ,val,c); }
  void min_val(int typ,double val,int c=1) { for (int ex=0;ex<evs.size();++ex) evs(ex)->min_val(typ,val,c); }
  void max_val(int typ,double val,int c=1) { for (int ex=0;ex<evs.size();++ex) evs(ex)->max_val(typ,val,c); }  
  
protected:
  void check_index(int index, const char *eyec) const {
    AzX::throw_if(index<0 || index>=evs.size(), eyec, "index is out of range"); 
  }  
}; 

/*----------------------------------------------------------------------*/
class AzpG_dg_info { /* for saving info needed for generation of warm-start */
protected:   
  static const int version = 1;  /* 2/5/2018 */
  static const int reserved_len=64;   
public: 
  bool do_pm1; 
  int minib; 
  AzpData_tmpl_save d_tmpl; 
  AzpDataRand z_data; 
  AzpG_dg_info() : do_pm1(false), minib(-1) {}
  AzpG_dg_info(bool _do_pm1, int _minib, const AzpData_tmpl_ &_d_tmpl, const AzpDataRand &_z_data) {
    do_pm1=_do_pm1; minib=_minib; d_tmpl.reset(_d_tmpl); z_data.copy_params_from(_z_data); 
  }
  bool is_set() const { return d_tmpl.is_set(); }
  void write(AzFile *file) const {
    AzX::throw_if(!is_set(), "AzpG_dg_info", "Not set"); 
    AzTools::write_header(file, version, reserved_len);       
    file->writeInt(minib); /* version > 0 */
    file->writeBool(do_pm1); d_tmpl.write(file); z_data.write(file); 
  }
  void read(AzFile *file) {
    int v = AzTools::read_header(file, reserved_len); 
    if (v > 0) minib = file->readInt(); 
    do_pm1 = file->readBool(); d_tmpl.read(file); z_data.read(file); 
  }
};

/*----------------------------------------------------------------------*/
/* D's and tilde{G} */
class AzpG_ddg {
  class AzpG_ddg_D {
  public: 
    AzpReNet *net; /* just pointing, not owning */ 
    double eta; 
    AzpG_ddg_D() : eta(0.1), net(NULL) {}    
    void write(AzFile *file) const {
      net->write(file); file->writeDouble(eta); 
    }
    void read(AzFile *file) {
      net->read(file); eta = file->readDouble(); 
    }
  };
protected:   
  AzDataArr<AzpG_ddg_D> d_nets;
  int num_in_file; /* input for write, output for read */
  static const int version = 1;   
  static const int reserved_len=64;   
public: 
  AzpReNet *g_net;  /* just pointing, not owning */   
  AzpG_ddg(AzpReNet *_g_net, AzObjPtrArr<AzpReNet> &_d_net) : num_in_file(-1) { 
    g_net=_g_net; 
    d_nets.reset(_d_net.size()); 
    for (int ix = 0; ix < d_nets.size(); ++ix) d_nets(ix)->net = _d_net(ix); 
    check(); 
  }  
  AzpG_ddg(AzpReNet *_g_net) : num_in_file(0) {
    g_net = _g_net; 
    num_in_file = 0; 
    d_nets.reset(); 
    check();     
  }   
  int get_num_in_file() const { return num_in_file; }
  void set_num_in_file(int _num) { num_in_file = _num; }   
  
  void write(AzFile *file, const AzpG_dg_info &info) const {
    const char *eyec = "AzpG_ddg::write";   
    AzX::throw_if(num_in_file<0, eyec, "num_in_file is not set."); 
    AzX::throw_if(num_in_file>d_nets.size(), eyec, "num_in_file is too large.");     
    
    AzTools::write_header(file, version, reserved_len);      
    file->writeInt(num_in_file);     
    info.write(file);     
    g_net->write(file); 
    for (int ix = 0; ix < num_in_file; ++ix) d_nets[ix]->write(file);     
  }
  void read(AzFile *file, AzpG_dg_info &info) {
    const char *eyec = "AzpG_ddg::read"; 
    check(); 
    int v = AzTools::read_header(file, reserved_len);     
    AzX::throw_if(v < 1, AzInputError, "AzpG_ddg::read", "version conflict"); 
    num_in_file = file->readInt();     
    info.read(file);      
    g_net->read(file); 
    AzX::throw_if(num_in_file>d_nets.size(), eyec, "the prepared d_nets is too small."); 
    for (int ix = 0; ix < num_in_file; ++ix) d_nets(ix)->read(file);     
  }
  static int read_num_in_file(const char *fn) {
    AzFile file(fn); file.open("rb"); 
    int v = AzTools::read_header(&file, reserved_len);
    AzX::throw_if(v < 1, AzInputError, "AzpG_ddg::read_num_in_file", "version conflict");     
    return file.readInt(); 
  }
  virtual void write_nets(const char *fn, const AzpG_dg_info &info) const { 
    AzFile file(fn); file.open("wb"); 
    write(&file, info); 
    file.close(true); 
  }
  virtual void read_nets(const char *fn, AzpG_dg_info &info) {
    AzFile file(fn); file.open("rb"); 
    read(&file, info); 
    file.close(); 
  }
  virtual void write_nets(const AzBytArr &s_fnstem, const AzpG_dg_info &info, 
                          const AzBytArr &s_clk, const AzOut &out) const { /* overriding */
    AzBytArr s_fn(&s_fnstem); 
    s_fn << s_clk << ".ddg"; 
    AzTimeLog::print("Saving networks to ", s_fn.c_str(), out); 
    write_nets(s_fn.c_str(), info); 
  }  
  
  int dNum() const { return d_nets.size(); }
  AzpReNet *d_net(int ix) { check_index(ix, "d_net"); return d_nets(ix)->net; }  
  double eta(int ix) const { check_index(ix, "eta");   return d_nets[ix]->eta; }
  
  void set_eta(double eta) { for (int ix=0;ix<d_nets.size();++ix) d_nets(ix)->eta = eta; }
  void set_eta(int ix, double eta) { check_index(ix, "set_eta"); d_nets(ix)->eta = eta; }
  
  void check() const {
    const char *eyec = "AzpG_ddg::check"; 
    AzX::throw_if_null(g_net, eyec, "g_net"); 
    for (int ix = 0; ix < d_nets.size(); ++ix) AzX::throw_if_null(d_nets[ix]->net, eyec, "d_net[ix]"); 
  }    
  void check_index(int idx, const char *msg) const {
    AzX::throw_if(idx<0 || idx>=d_nets.size(), "AzpG_ddg", msg); 
  }                          
}; 

/*----------------------------------------------------------------------*/
class AzpG_Cfg_G {
public:   
  virtual void apply_G0(AzpReNet *g_net, bool is_test, const AzPmatVar &mvi, AzPmatVar &mvo) const = 0; 
  virtual void apply_gt(AzPmatVar &mv_fake, /* inout */
                   AzpReNet *d_net, double eta, 
                   const AzPmatVar &mv_z) const = 0;  
}; 

/*----------------------------------------------------------------------*/
/* training data for updating the approximator */
class AzpG_xy_ {
public: 
  virtual int dataNum() const = 0; /* number of data points */
  virtual void input(const AzIntArr &ia_dxs, AzPmatVar &mv) const = 0;
  virtual void output(const AzIntArr &ia_dxs, AzPmatVar &mv) const = 0;
  virtual void input_output(const AzIntArr &ia_dxs, AzPmatVar &mvi, AzPmatVar &mvo) const {
    input(ia_dxs, mvi); 
    output(ia_dxs, mvo); 
  }
};

/*----------------------------------------------------------------------*/
class AzpG_xy0 : public virtual AzpG_xy_ {
protected: 
  AzPmatVarData mvd_x, mvd_y; 
public: 
  AzpG_xy0() {}
  void init(const AzPmatVar &mv_x_tmpl, const AzPmatVar &mv_y_tmpl, int num) {
    mvd_x.init_for_put_seq(mv_x_tmpl, num); 
    mvd_y.init_for_put_seq(mv_y_tmpl, num); 
  }
  void put(int dx, const AzPmatVar &mv_x, const AzPmatVar &mv_y) {
    mvd_x.put_seq(dx, mv_x); 
    mvd_y.put_seq(dx, mv_y);     
  }
  int dataNum() const { return mvd_x.dataNum(); }
  void input( const AzIntArr &ia_dxs, AzPmatVar &mv) const { mvd_x.get(ia_dxs, mv); }
  void output(const AzIntArr &ia_dxs, AzPmatVar &mv) const { mvd_y.get(ia_dxs, mv); }
}; 

/*----------------------------------------------------------------------*/
class AzpG_Pool : public virtual AzpG_xy_ {
protected: 
  AzpG_Cfg_G *G; 
  virtual void check_G(const char *eyec) const { AzX::throw_if_null(G, eyec, "G"); }
  AzPmatVar mv_g_tmpl; 
  
  AzIntArr ia_dseq; int pos_in_dseq;
  
  AzPmatVarData mvd_z; /* input */
  AzPmatVarData mvd_g; /* generated */
  bool doing_pool() const { return (mvd_g.dataNum() > 0); }
  AzIntArr ia_tt; 
  AzRandGen rg; 
  AzOut out; 

  bool do_chk; 
  
  virtual void _reset_tt(int data_num) {
    ia_tt.reset(data_num, -1); 
    _reset_dseq(data_num); 
  }
  virtual void _reset_dseq(int data_num) {
    ia_dseq.range(0, data_num); pos_in_dseq = data_num;
  }
  virtual void _init_g_pool(int data_num) {
    AzX::throw_if(mv_g_tmpl.dataNum() <= 0, "AzpG_Pool::_init_g_pool", 
                  "No template for generated data"); 
    dev.show_mem_stat(log_out, "_init_g_pool begins: "); 
    mvd_g.init_for_put_seq(mv_g_tmpl, data_num);  
    dev.show_mem_stat(log_out, "_init_g_pool ends: ");     
  }
public:  
  virtual void reset_all(const AzpDataRand *f_data, AzpReNet *g_net, int data_num, int minib, 
                      bool do_nopl=false) {
    AzX::throw_if_null(g_net, f_data, "AzpG_Pool::reset_all");   
    AzTimeLog::print("Refreshing ... ", data_num, out); 
    mvd_z.reset(); mvd_g.reset();     
    _reset_tt(data_num);    
    for (int dx = 0; dx < data_num; dx += minib) {
      int d_num = MIN(minib, data_num-dx); 
      bool is_test = true;       
      AzPmatVar mv_z; f_data->gen_data(d_num, mv_z);
      if (dx == 0) {
        if (mv_g_tmpl.dataNum() <= 0) {
          AzPmatVar mv_z_tmpl; mv_z_tmpl.set(&mv_z, 0, 1); 
          check_G("AzpG_Pool::_reset"); 
          G->apply_G0(g_net, is_test, mv_z_tmpl, mv_g_tmpl);   
        }
        mvd_z.init_for_put_seq(mv_z, data_num); 
        if (!do_nopl) _init_g_pool(data_num);       
      }
      mvd_z.put_seq(dx, mv_z);
    }
  }    
  AzpG_Pool(AzpG_Cfg_G *_G, const AzOut &_out, int rand_seed=-1) : G(NULL), pos_in_dseq(-1), do_chk(false) { 
    G = _G;
    check_G("AzpG_Pool()"); 
    out = _out; rg._srand_(rand_seed); 
  }
  void reset_do_chk(bool _do_chk) { do_chk = _do_chk; }   
  virtual void remove_output() {
    mvd_g.reset(); 
    _reset_tt(dataNum()); 
  }  
  virtual int dataNum() const { return mvd_z.dataNum(); }  
  virtual void output(const AzIntArr &ia_dxs, AzPmatVar &mv_data) const { 
    const char *eyec = "AzpG_Pool::output"; 
    check_pool(eyec);     
    for (int ix = 0; ix < ia_dxs.size(); ++ix) {
      AzX::throw_if(ia_tt[ia_dxs[ix]] < 0, eyec, "requested data is uninitialized."); 
    }
    mvd_g.get(ia_dxs, mv_data); 
  }
  virtual void input(const AzIntArr &ia_dxs, AzPmatVar &mv_z) const { mvd_z.get(ia_dxs, mv_z); }  
  
  virtual void pick_data(bool is_g_test, int tt1, AzpG_ddg &tns, int minib, AzPmatVar &mv, int test_minib, 
                 AzPmatVar *mv_z=NULL) {
    const char *eyec = "AzpG_Pool::pick_data_dseq"; 
    AzX::throw_if(minib > ia_tt.size(), eyec, "minib > ia_tt.size()"); 
    AzX::throw_if(pos_in_dseq < 0 || pos_in_dseq > ia_tt.size() || ia_dseq.size() != ia_tt.size(), 
                  eyec, "data sequence is not ready"); 
    
    if (pos_in_dseq >= ia_dseq.size()) { /* generate a new sequence if the end of data sequence is reached */
      if (doing_pool()) catchup(is_g_test, tt1, tns, minib);     
      AzTools::shuffle2(ia_dseq, &rg); 
      AzPrint::writeln(out, eyec, " Generating a new sequence ... ", ia_tt[ia_dseq[0]]); 
      pos_in_dseq = 0;       
    }
    
    int tt0 = ia_tt[ia_dseq[pos_in_dseq]]; 
    AzX::throw_if(tt0 < -1 || tt0 > tt1, eyec, "Unexpected tt"); 

    AzIntArr ia_dxs; 
    for (int ix = 0; ix < minib; ++ix, ++pos_in_dseq) {
      if (pos_in_dseq >= ia_dseq.size()) break; 
      ia_dxs.put(ia_dseq[pos_in_dseq]); 
    }
    if (ia_dxs.size() < minib) AzPrint::writeln(out, "Warning: #picked=", ia_dxs.size()); 
    refine(is_g_test, ia_dxs, tt0, tt1, tns, mv);
    if (mv_z != NULL) input(ia_dxs, *mv_z); 
  } 
 
public: 
  virtual void catchup(bool is_g_test, int tt1, AzpG_ddg &tns, int minib) {
    const char *eyec = "AzpG_Pool::catchup"; 
    bool be_silent = false; 
    int data_num = dataNum();    
    if (!doing_pool()) _init_g_pool(data_num);
    int inc = data_num / 50, milestone = inc;     
    for (int dx = 0; dx < data_num; ++dx) {
      if (!be_silent) AzTools::check_milestone(out, milestone, dx, inc);       
      int tt0 = ia_tt[dx]; 
      AzX::throw_if(tt0>tt1, eyec, "tt0>tt1?!"); 
      if (tt0 == tt1) continue; 
      AzIntArr ia_dxs; 
      for (int xx = dx; xx < data_num; ++xx) {
        if (ia_tt[xx] != tt0) continue; 
        ia_dxs.put(xx); 
        if (ia_dxs.size() >= minib) break;         
      }
      AzPmatVar mv; 
      refine(is_g_test, ia_dxs, tt0, tt1, tns, mv); 
    }
    if (!be_silent) AzTools::finish_milestone(out, milestone);    
    AzX::throw_if(ia_tt.min() != tt1 || ia_tt.max() != tt1, eyec, "didn't catch up?!"); 
    _reset_dseq(data_num);    
  }    
protected:
  virtual void refine(bool is_g_test, const AzIntArr &ia_dxs, int tt0, int tt1, AzpG_ddg &tns, 
              AzPmatVar &mv) {
    check_G("AzpG_Pool::refine");                
    AzX::throw_if(tt0<-1, "AzpG_Pool::refine", "tt0<-1?!");                 
    if (do_chk) {
      for (int ix = 0; ix < ia_dxs.size(); ++ix) {
        AzX::throw_if(ia_tt[ia_dxs[ix]] != tt0, "AzpG_Pool::refine", "Expected tt=tt0"); 
      }      
    }
    AzPmatVar mv_z; mvd_z.get(ia_dxs, mv_z);
    if (tt0 == -1) {                      
      G->apply_G0(tns.g_net, is_g_test, mv_z, mv);
    }
    else {
      check_pool("AzpG_Pool::refine"); 
      mvd_g.get(ia_dxs, mv); /* get data */
    }
    for (int tx = MAX(0, tt0); tx < tt1; ++tx) {
      G->apply_gt(mv, tns.d_net(tx), tns.eta(tx), mv_z); /* refine */
    }
    if (doing_pool()) {
      mvd_g.put(ia_dxs, mv);  /* put the refined data back */
      for (int ix = 0; ix < ia_dxs.size(); ++ix) {
        ia_tt(ia_dxs[ix], tt1);  /* record that their refinement level is now tt1 */
      }  
    }
  }
  virtual void check_pool(const char *eyec) const {
    AzX::throw_if(!doing_pool(), eyec, "check_pool: No pool"); 
  }  
};             

class AzpG_stat {
public:
  double tune_tim, tune_clk, apprx_tim, apprx_clk; 
  AZint8 g_upd, d_upd; 
  AzpG_stat() : tune_tim(0), tune_clk(0), apprx_tim(0), apprx_clk(0), g_upd(0), d_upd(0) {}
  void reset() { tune_tim=tune_clk=apprx_tim=apprx_clk=0; g_upd=d_upd=0; }
  double clk() const { return tune_clk+apprx_clk; }
  double tim() const { return tune_tim+apprx_tim; }
};              

class AzpG_alarm {
protected: 
  double inc, target, last_ring_target; 
public: 
  AzpG_alarm(double _inc) : inc(-1), target(-1), last_ring_target(-1) {
    if (_inc <= 0) return; 
    inc = target = _inc;
  }
  bool is_ringing(double clk) { 
    if (inc <= 0) return false; 
    bool ringing = (clk >= target); 
    if (ringing) {
      last_ring_target = target; 
      while(target <= clk) target += inc;   
    }
    return ringing; 
  } 
  double last_ring() const { return last_ring_target; }
  void turn_off() { inc = -1; }
  bool is_on() const { return (inc > 0); }
}; 

class AzpG_Gen {
public:  
  virtual void generate(AzpG_ddg &ddg, const AzpG_dg_info &info, 
                AzParam &g_azp, AzParam &d_azp, AzParam &azp) = 0; 
};
                
/*----------------------------------------------------------------------*/
class AzpG_Cfg : public virtual AzpG_Cfg_G, public virtual AzpG_Gen {
protected: 
  /******    cfg (xICFG)  ******/
  double cfg_eta;  
  int cfg_T, cfg_U; 
  int cfg_N; /* # of data points used for updating the approximator */
  double cfg_diff_max; 
  int approx_redmax, approx_epo; 
  double approx_decay, rd_wini;  
  bool doing_randinit() const { return (rd_wini > 0); }
  int poolsz; bool do_reuse_pool; 
  
  /******  general  ******/
  AzOut out, less_out, eval_out; 
  bool do_gtr; 
  AzBytArr s_gen_fn; 
  int gen_num;   
  bool do_pm1, do_no_collage; 
  AzRandGen rg;
  virtual void init_rg_if_required() {
    if (rg.doing_myown()) return;
  }
  void check_rg(const char *msg) const {}

  AzpDataSeq dataseq;  /* data sequencer to make mini-batches */
  int rseed; 
  int test_num, inc; 
  int minib, test_minib; 
  AzBytArr s_save_fn;
  int tst_seed, gen_seed; 
  bool do_reset_gen_seed; 
  double clk_max, clk_min, test_clk, save_clk, gen_clk; 
  bool do_verbose; 
  double ss_decay, ss_clk; 

  /*---  used for maintenance purposes only  ---*/
  int trn_seed; 

public:
  AzpG_Cfg() : out(log_out), 
       /***** cfg parameters *****/
       cfg_eta(-1), cfg_U(1), cfg_T(25), cfg_N(640), 
       /***** cfg constants (that should be fixed to these values typically) *****/
       cfg_diff_max(40), approx_redmax(3), approx_decay(0.1), rd_wini(0.01),
       approx_epo(10), poolsz(-1), do_reuse_pool(false), 
       /***** more general parameters and constants *****/ 
       do_pm1(false), gen_num(-1), do_no_collage(false), 
       rseed(1), test_num(10000), inc(1000), minib(64), test_minib(-1), 
       tst_seed(9), gen_seed(8), do_reset_gen_seed(true),              
       clk_max(100000), clk_min(0), test_clk(1000), save_clk(-1), gen_clk(-1), 
       do_verbose(false), ss_clk(-1), ss_decay(0.1), do_gtr(false), trn_seed(-1)
       {}
  virtual void cfg_train(const AzOut *_eval, AzpG_ddg &ddg, const AzpG_dg_info *infp, const AzpDataSet_ &r_ds, 
                 AzParam &g_azp, AzParam &d_azp, AzParam &azp, AzpG_ClasEval_ &cevs); 

  virtual void generate(AzpG_ddg &ddg, const AzpG_dg_info &info, 
                AzParam &g_azp, AzParam &d_azp, AzParam &azp); 
                 
protected:
  /******    cfg (xICFG)   ******/
  virtual void resetParam_cfg(AzParam &azp);                         
  virtual int g_config(AzpG_ddg &ddg, AzParam &g_azp, 
                          const AzpData_tmpl_ &z_trn, const AzpData_tmpl_ &r_trn); 
  virtual void d_g_init(AzpReNet *d_net, int ini_tt, AzpG_ddg &ddg, AzParam &g_azp, 
                      const AzpDataRand &z_trn, const AzpData_ &r_trn, AzpG_stat &st, 
                      AzpG_Pool &pltst, AzpG_ClasEval_ &cevs); 
  virtual void d_config(AzpReNet *d_net, AzpG_ddg &ddg, AzParam &d_azp, 
                          int init_tt, const AzpData_tmpl_ &r_trn); 
  virtual void update_G(AzpG_ddg &ddg, int tt, const AzpReNet *d_net, AzParam &d_azp, 
                        const AzpData_tmpl_ &r_trn); 
  
  virtual void inline apply_G0(AzpReNet *g_net, bool is_test, const AzPmatVar &mvi, AzPmatVar &mvo) const {
    g_net->up(is_test, mvi, mvo);  
  }
  virtual void apply_gt(AzPmatVar &mv_fake, /* inout */
                   AzpReNet *d_net, double eta, const AzPmatVar &mv_z) const;
  virtual void inline gen_last_step(AzPmatVar &mv_fake, bool is_trn=false) const { 
    if (!is_trn) truncate(*mv_fake.data_u()); 
  }   
  virtual void inline truncate(AzPmat &m_fake) const {  
    if (do_pm1) m_fake.truncate(-1,1); 
  }  
   
  /*---  check the progress  ---*/
  virtual bool check_progress(int tt, const AzpG_stat &st, AzpG_Ev &ev) const; 
   
  /*---  train the approximator  ---*/
  virtual void approximate(AzpG_ddg &ddg, int tt, AzpG_Pool &pltrn, int x_epo, double ini_ccoeff=1); /* const (rg) */ 
  virtual void approx(AzpReNet *g_net, AzpG_xy_ &pltrn, int x_epo, double ini_ccoeff=1); /* const (rg) */   
  virtual bool chk_loss(AzpReNet *net, AzBytArr &s_loss, int ite, double &loss, 
                double &count, double &last_loss_avg, 
                int &max_reduce, int &reduce_count, double &ccoeff) const;                  
                   
  /*---  initialize the approximator  ---*/
  virtual void random_init(AzpReNet *g_net, const AzpData_tmpl_ &d_tmpl, const AzpDataRand &z_trn); 
  virtual void gen_rand(int d_num, int cc, int sz, 
                const AzpDataRand &z_trn,
                const AzPmat &m_w, 
                AzPmatVar &mvo, /* output */
                AzPmatVar *mv_z=NULL) const;   
                
  /*---  test and saving  ---*/              
  virtual void cfg_test(const AzpG_stat &stat, 
                AzpG_ddg &ddg, int tt, 
                const AzpG_Ev &ev,  
                AzpG_Pool &pltst, 
                AzpG_ClasEval_ &cevs) const; 
  virtual void save_ddg(const AzpG_dg_info &info, int tt, double clk, AzpG_ddg &ddg); 
 
  /******     more general     ******/
  virtual int loop_d_update(int my_U, AzpReNet *d_net, const AzpData_ *r_trn, const int *dxs, int ini_index,
                AzpG_Pool &pltrn, AzpG_ddg &ddg, int tt, AzpG_stat &st, AzpG_Ev_ &evarr); 
  virtual void d_update(AzpReNet *d_net, const AzpData_ *r_trn, const int *dxs, int index,
                AzpG_Pool &pltrn, AzpG_ddg &ddg, int tt, AzpG_Ev_ &evarr);  
  virtual void resetParam_general(AzParam &azp); 
  virtual void check_input(const AzpDataSet_ &r_ds, const AzpG_dg_info *infp=NULL); 
  virtual void init_rand(AzParam &azp, AzpDataRand &z_trn, AzpDataRand &z_tst, AzpDataRand &z_gen, 
                         const AzpG_dg_info *infp=NULL) const; 
                       
  /*---  feed real data to D for fprop  ---*/                                               
  virtual void real_up(bool is_test, AzpReNet *d_net, const AzpData_ *r_data, 
                       const int *dxs, int d_num, AzPmatVar &mv, 
                       AzpG_Ev_ &evarr,
                       AzPmatVar *mv_real=NULL); 

  /*---  get the loss derivatives  ---*/
  virtual void deriv_real(AzPmatVar &mv, AzpG_Ev_ &evarr) { deriv(true, mv, evarr); } 
  virtual void deriv_fake(AzPmatVar &mv, AzpG_Ev_ &evarr) { deriv(false, mv, evarr); }
  virtual void deriv(bool is_real, AzPmatVar &mv, AzpG_Ev_ &evarr); 
  virtual void logi(double yy, AzPmatVar &mv) const { logi(yy, *mv.data_u()); }
  virtual void logi(double yy, AzPmat &m) const; 
  
  /*---  copy networks  ---*/              
  virtual void copy_net(AzpReNet *out_net, const AzpReNet *inp_net, 
                AzParam &azp, const AzpData_tmpl_ *trn,
                bool do_show=false,
                bool for_testonly=false) const; 
                
  /*---  evaluate generated images  ---*/              
  virtual void eval_gen(AzBytArr &s, AzpG_ddg &ddg, int tt, AzpG_ClasEval_ &cevs, AzpG_Pool &pldata) const; 
  virtual void format_eval(AzBytArr &s, const AzpG_Ev &ev) const {
    ev.format_eval(s, do_verbose); 
  }
  virtual void show_eval(const AzpG_Ev &ev) const {  
    AzBytArr s; 
    format_eval(s, ev); 
    AzPrint::writeln(out, s); 
  }  
    
  /*---  ---*/                
  virtual bool check_img_range(const AzPmatVar &mv_img, const char *msg, bool dont_throw=false) const; 
  virtual bool check_fake_img_range(AzpReNet *g_net, const AzpG_Pool &pltst, 
                            const char *msg, bool dont_throw=false) const; 

  /*---  for generation  ---*/ 
  virtual void gen_in_trn(AzpG_ddg &ddg, const AzpData_tmpl_ &d_tmpl, AzpDataRand &z_gen, int tt, double clk) const; 
  virtual void gen(int tt, const AzpDataRand &z_gen, int d_num, AzpG_ddg &ddg, AzPmatVar &mv_gen) const; 
  virtual void generate_bin(AzpG_ddg &ddg, const AzpData_tmpl_ &d_tmpl, 
                            const AzpDataRand &z_gen, int tt) const; 
  virtual void generate_ppm(AzpG_ddg &ddg, const AzpData_tmpl_ &d_tmpl, 
                    const AzpDataRand &z_gen, int tt, const char *clk_str=NULL) const; 
 
  /*---  ---*/  
  virtual int get_size(const AzpData_tmpl_ &d_tmpl) const;
  static void clk_str(double clk, AzBytArr &s_clk) {
    if (clk >= 0) {
      int width = 6; bool fill_zero = true;
      s_clk << "-clk";
      s_clk.concatInt((int)clk, width, fill_zero);
    }
  }    
};
#endif 