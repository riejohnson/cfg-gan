/* * * * *
 *  AzpG_Cfg.cpp 
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
#include "AzpG_Cfg.hpp"
#include "AzpData_imgbin.hpp"  /* for kw_do_pm and kw_do_01 */
#include "AzpG_Tools.hpp"

static const int evshrt = 0, evlng = 1; /* short range, long range */ 

/*------------------------------------------------------------*/ 
#define kw_minib       "mini_batch_size="      /* mini batch size for training      */
#define kw_test_minib  "test_mini_batch_size=" /* mini batch size for testing       */
#define kw_rseed       "random_seed="          /* seed for random number generation */
#define kw_tst_seed    "tst_seed=" /* random number seed for test data generation   */
#define kw_gen_seed    "gen_seed=" /* random number seed for generating images      */
#define kw_do_reset_gen_seed "ResetGenSeed"
#define kw_dont_reset_gen_seed "DontResetGenSeed"
#define kw_inc "inc="
#define kw_do_verbose "Verbose"
#define kw_test_num "num_test=" /* number of generated images to be tested         */
#define kw_save_fn "save_fn="   /* used for generating filenames for saving models */
#define kw_clk_max  "max_clk="  /* maximum time for training */
#define kw_clk_min  "min_clk="  /* minimum time for training */
#define kw_save_clk "save_clk=" /* e.g., if this is 10000, save a model after every 10K seconds  */
#define kw_test_clk "test_clk=" /* e.g., if this is 1000, evaluate images after every 1K seconds */
#define kw_gen_clk  "gen_clk="  /* e.g., if this is 1000, generate images after every 1K seconds */
#define kw_gen_fn "gen_fn="     /* used for generating filenames for saving generated images     */
#define kw_gen_num "num_gen="   /* number of images to be genrated */
#define kw_do_no_collage "NoCollage" /* generate individual ppm/pgm files instead of making collages */
#define kw_ss_decay "ss_decay=" /* reduce step-size for D and tilde{G} by multiplying this value */
#define kw_ss_clk "ss_clk="     /* e.g., if this is 100000, reduce step-size for D and tilde{G} */
                                /* after every 100K seconds */
#define kw_do_gtr "Gtr"                                      
/*------------------------------------------------------------*/ 
void AzpG_Cfg::resetParam_general(AzParam &azp) {
  const char *eyec = "AzpG_Cfg::resetParam_general"; 
  AzPrint o(out); 
  azp.swOn(o, do_pm1, kw_do_pm1); 
  bool do_01=false; azp.swOn(o, do_01, kw_do_01);
  AzX::no_support(do_01, eyec, kw_do_01); 
  
  azp.vInt(o, kw_rseed, rseed); 
  azp.vInt(o, kw_minib, minib);   
  azp.vInt(o, kw_test_minib, test_minib); 
  azp.vInt(o, kw_test_num, test_num);    
  azp.vStr(o, kw_save_fn, s_save_fn); 
  azp.vInt(o, kw_inc, inc); 

  dataseq.resetParam(azp); /* for mini-batch generation */
  dataseq.printParam(out); 
     
  azp.vFloat(o, kw_test_clk, test_clk); 
  azp.vFloat(o, kw_save_clk, save_clk); 
  azp.vFloat(o, kw_gen_clk, gen_clk);   
  azp.vFloat(o, kw_clk_max, clk_max);  
  azp.vFloat(o, kw_clk_min, clk_min);    
  
  azp.vInt(o, kw_tst_seed, tst_seed);   
  azp.vInt(o, kw_gen_seed, gen_seed);    
  azp.swOn(&do_reset_gen_seed, kw_do_reset_gen_seed);      
  azp.swOff(&do_reset_gen_seed, kw_dont_reset_gen_seed, kw_do_reset_gen_seed);      
  o.printSw(kw_do_reset_gen_seed, do_reset_gen_seed);   
  azp.swOn(o, do_verbose, kw_do_verbose); 
  azp.vInt(o, kw_gen_num, gen_num); 
  azp.vStr(o, kw_gen_fn, s_gen_fn); 
  if (gen_num > 0 && s_gen_fn.length() > 0) {
    azp.swOn(o, do_no_collage, kw_do_no_collage); 
  }
  azp.swOn(o, do_gtr, kw_do_gtr); 
  
  azp.vFloat(o, kw_ss_clk, ss_clk); 
  if (ss_clk > 0) {
    azp.vFloat(o, kw_ss_decay, ss_decay); 
    AzXi::throw_if_nonpositive(ss_decay, eyec, kw_ss_decay); 
  }
  
  AzXi::throw_if_nonpositive(minib, eyec, kw_minib);   
  if (test_minib <= 0) test_minib = minib;  
  AzXi::throw_if_nonpositive(clk_max, eyec, kw_clk_max); 
}

/*------------------------------------------------------------*/ 
#define kw_poolsz       "cfg_pool_size=" /* pool size */
#define kw_cfg_U        "cfg_U="         /* U: discriminator update frequency */
#define kw_cfg_T        "cfg_T="         /* T */
#define kw_cfg_eta      "cfg_eta="       /* eta for G_t(x) = G_{t-1}(x) + eta g_t(G_{t-1}(x))*/
#define kw_approx_epo   "cfg_x_epo="     /* number of epochs for updating the approximator */

/*---  These should be fixed to the default values, typically.  ---*/
#define kw_approx_redmax "cfg_x_redmax=" /* for updating approximator tilde{G}             */
                                         /* e.g., if this is 3, reduce the step-size up to */
                                         /* 3 times and stop iterating on the 4th request  */
#define kw_approx_decay "cfg_x_decay="   /* for updating approximator tilde{G}             */
                                         /* e.g., if this is 0.1, reduce the step-size by  */
                                         /* multipling 0.1 when the loss is not going down */
#define kw_cfg_diff_max "cfg_diff_max="  /* stop training when |D(real)-D(gen)| exceeds this */
#define kw_rd_wini      "cfg_rand_wini="
/*------------------------------------------------------------*/ 
void AzpG_Cfg::resetParam_cfg(AzParam &azp) {
  const char *eyec = "AzpG_Cfg::resetParam_cfg";  
  resetParam_general(azp);  
  AzPrint o(out); 
 
  azp.vInt(o, kw_cfg_T, cfg_T);  /* T */
  azp.vInt(o, kw_poolsz, poolsz); /* pool size */  
  azp.vInt(o, kw_cfg_U, cfg_U);  /* U */   
  azp.vFloat(o, kw_cfg_eta, cfg_eta);  /* step-size in ICFG */
   
  /*---  use default values for these, typically  ---*/
  azp.vInt(o, kw_approx_redmax, approx_redmax); 
  azp.vFloat(o, kw_approx_decay, approx_decay); 
  azp.vFloat(o, kw_rd_wini, rd_wini); 
  azp.vFloat(o, kw_cfg_diff_max, cfg_diff_max); 
  azp.vInt(o, kw_approx_epo, approx_epo); 

  /*---  check values  ---*/
  if (poolsz < minib || poolsz % minib != 0) {
    AzBytArr s(kw_poolsz); s << " must be a multiple of " << kw_minib << "."; 
    AzX::throw_if(true, AzInputError, eyec, s.c_str()); 
  }
  AzXi::throw_if_nonpositive(kw_cfg_U, eyec, kw_cfg_U); 
  AzXi::throw_if_nonpositive(cfg_eta, eyec, kw_cfg_eta);    
  AzXi::throw_if_nonpositive(cfg_T, eyec, kw_cfg_T);   
  AzXi::throw_if_nonpositive(poolsz, eyec, kw_poolsz); 
  AzXi::throw_if_nonpositive(approx_epo, eyec, kw_approx_epo);  
  AzX::throw_if(approx_decay<0 || approx_decay>=1, eyec, kw_approx_decay, " must be in (0,1).");    
}

/*------------------------------------------------------------*/ 
void AzpG_Cfg::cfg_train(const AzOut *_eval, 
                         AzpG_ddg &ddg, /* D_0,D_1,...,D_T,tilde{G} */
                         const AzpG_dg_info *infp, /* used for warm-start */
                         const AzpDataSet_ &r_ds,  /* real data */
                         AzParam &g_azp, /* net config for tilde{G} */
                         AzParam &d_azp, /* net config for D */
                         AzParam &azp,   /* parameters */
                         AzpG_ClasEval_ &cevs) { /* for evaluation using a classifier */
  const char *eyec = "AzpG_Cfg::cfg_train";
  if (_eval != NULL) eval_out = *_eval;  

  resetParam_cfg(azp);  /* parse parmeters */
  if (rseed > 0) srand(rseed);      /* reset random number generator */   
  int ini_tt = MAX(0, ddg.get_num_in_file());   
  if (ini_tt > 0 && rd_wini > 0) {
    AzPrint::writeln(out, kw_rd_wini, " is not used as warm-starting from a saved model."); 
    rd_wini = -1; /* this needs to be done before calling init_rg_if_required */
  }   
  init_rg_if_required();    

  /*---  check input  ---*/
  ddg.check(); check_input(r_ds, infp);   
  const AzpData_ *r_trn = r_ds.trn_data(), *r_tst = r_ds.tst_data(); 
  AzpDataRand z_trn, z_tst, z_gen; 
  init_rand(azp, z_trn, z_tst, z_gen, infp); 
  azp.check(out);
  
  if (do_verbose) less_out = out; 

  /*---  configure approximator tilde{G} (g_net) ---*/
  AzPrint::writeln(out, "=====  g_net  =====");       
  int g_outdim = g_config(ddg, g_azp, z_trn, *r_trn); 
  
  /*---  configure discriminator D (d_net) ---*/
  AzPrint::writeln(out, "=====  d_net  ====="); 
  AzObjPtrArr<AzpReNet> darr; 
  AzpReNet *d_net = ddg.d_net(0)->safe_clone_nocopy(darr); 
  d_config(d_net, ddg, d_azp, ini_tt, *r_trn); 
  
  AzPrint::writeln(out, "g:outdim=", g_outdim);
  AzX::throw_if(ddg.dNum() < cfg_T, AzInputError, eyec, "ddg.dNum() < cfg_T"); 
  
  /*---  configure the evaluator using a classifier  ---*/
  cevs.init(less_out, r_trn, r_tst, test_minib);
    
  /*---  set up data  ---*/
  dataseq.init(r_ds.trn_data(), out); 
  (const_cast<AzpData_ *>(r_trn))->first_batch(); 
  AzpG_Pool pltrn(this, less_out); /* for training */
  AzpG_Pool pltst(this, less_out); /* for testing */
  pltrn.reset_all(&z_trn, ddg.g_net, poolsz, test_minib); 
  pltst.reset_all(&z_tst, ddg.g_net, test_num, test_minib); 
  check_fake_img_range(ddg.g_net, pltst, eyec);   

  /*---  initialize timers, statistics, and input info  ---*/
  AzpG_alarm save_alarm(save_clk), gen_alarm(gen_clk), test_alarm(test_clk); 
  AzpG_alarm ss_alarm(ss_clk); double ss_factor = 1; 
  AzpG_stat st;  /* for keeping statistics */
  AzpG_dg_info info(do_pm1, test_minib, *r_trn, z_trn); /* info on input used when saving models */
  
  /*---  initialize the approximator by training for random images  ---*/
  d_g_init(d_net, ini_tt, ddg, g_azp, z_trn, *r_trn, st, pltst, cevs); 

  /*---  training  ---*/
  AzTimeLog::print("Training loop ... ", out);   
  int data_size = r_trn->dataNum();  /* size of this data batch */
  AzX::throw_if(data_size < minib*cfg_U, AzInputError, eyec, "data_size < minib*cfg_U");   
  const int *dxs = dataseq.gen(r_trn, data_size); /* a random sequence of data indexes */
  int index = 0; /* index indicating a position in dxs */
  AzpG_EvArray<AzpG_Ev> evarr(2);   
  for (int tt = ini_tt; ; ++tt) { /* ini_tt=0 if this is cold-start */
    AzBytArr s("***************************  t="); s << tt << ", d_upd=" << st.d_upd;   
    AzPrint::writeln(less_out, s);         

    if (tt>=cfg_T) { /* approximation is needed? */
      /*---  check progress, test, save, and generate if needed  ---*/        
      bool do_exit = check_progress(tt, st, *evarr(evshrt)); 
      if (do_exit) break;  /* exit as it's failing */

      double clk = st.clk();
      if (test_alarm.is_ringing(clk)) { 
        cfg_test(st, ddg, tt, *evarr[evlng], pltst, cevs); /* test */    
        evarr(evlng)->reset(); 
      }
      if (gen_alarm.is_ringing(clk)) gen_in_trn(ddg, info.d_tmpl, z_gen, tt, gen_alarm.last_ring());
      if (save_alarm.is_ringing(clk)) save_ddg(info, tt, save_alarm.last_ring(), ddg);     
      if (clk_max > 0 && clk > clk_max) {      
        AzTimeLog::print("Reached the maximum time ... exiting ... ", out); 
        if (save_alarm.is_on() && clk_max > save_alarm.last_ring()) save_ddg(info, tt, clk_max, ddg);        
        break;  /* exit if max time was reached */
      }       
      if (ss_alarm.is_ringing(clk)) {
        ss_factor *= ss_decay; 
        d_net->multiply_to_stepsize(ss_factor); 
        AzTimeLog::print("Changing step-sizes for D and tilde{G} to the initial value times ", ss_factor, out); 
      }

      /***  update approximator tilde{G} (g_net)  ***/      
      AzClock myclk(true); 
      approximate(ddg, tt, pltrn, approx_epo, ss_factor); 
      pltrn.reset_all(&z_trn, ddg.g_net, poolsz, test_minib); /* training data: refresh the input pool */
      pltst.reset_output_only();  /* test data: keep the input and remove the generator output */
      tt = 0;           
      myclk.tick(st.apprx_clk, st.apprx_tim);        
    } /* approximation is needed? */
   
    if (index + cfg_U*minib > data_size) { /* we hit the end of this data batch? */
      (const_cast<AzpData_ *>(r_trn))->next_batch(); /* load the next data batch */
      data_size = r_trn->dataNum();  /* size of this data batch */
      AzX::throw_if(data_size < cfg_U*minib, AzInputError, eyec, "data_size < cfg_U*minib"); 
      dxs = dataseq.gen(r_trn, data_size); 
      index = 0;      
    }
      
    /*---  update discriminator D (d_net) ---*/
    index = loop_d_update(cfg_U, d_net, r_trn, dxs, index, pltrn, ddg, tt, st, evarr); 
    if (evarr[evshrt]->is_nan()) {
      AzPrint::writeln(out, "Detected nan ... "); 
      break;         
    }    
    
    /*---  show progress ---*/
    s.reset("clk,"); s << st.clk(); 
    if (do_verbose) s << ",tim," << st.tim(); 
    format_eval(s, *evarr[evshrt]); AzPrint::writeln(less_out, s);
    
    /*---  update generaotr G by adding a discriminator to ddg  ---*/
    update_G(ddg, tt, d_net, d_azp, *r_trn); 
    ++st.g_upd;   
  } /* tt */
}

/*------------------------------------------------------------*/ 
/* fake += eta*g_t(fake) */
void AzpG_Cfg::apply_gt(AzPmatVar &mv_fake, /* inout */
                        AzpReNet *d_net, double eta, 
                        const AzPmatVar &mv_z) /* not used */ const {
  bool is_test = false;                            
  AzPmatVar mv; 
  d_net->up(is_test, mv_fake, mv); /* fprop */
  mv.data_u()->set(1);
  bool dont_update = true;
  d_net->down(mv, dont_update);    /* bprop */
  d_net->lay0()->get_ld(-1, mv);   /* mv <- nabla D */
  d_net->release_ld();
  mv_fake.data_u()->add(mv.data(), eta); /* mv_fake += eta*mv */
}

/*------------------------------------------------------------*/ 
/* input: real data x, output: derivatives of loss w.r.t. D(x) */
void AzpG_Cfg::real_up(bool is_test, 
                  AzpReNet *d_net, const AzpData_ *r_data, 
                  const int *dxs, int d_num, /* data indexes of a mini batch */
                  AzPmatVar &mv, /* output: derivatives of loss w.r.t. D(x) */
                  AzpG_Ev_ &evarr,
                  AzPmatVar *mv_real) /* optional output */ {
  const char *eyec = "AzpG_Cfg::real_up";                     
  AzX::throw_if_null(r_data, eyec, "r_data");                     
  AzPmatVar mv0; r_data->gen_data(dxs, d_num, mv0); 
  if (mv_real != NULL) mv_real->set(&mv0); 
  d_net->up(is_test, mv0, mv); /* fprop */
  AzX::throw_if(mv.colNum() != d_num, eyec, "More than one output per data");   
  deriv_real(mv, evarr);   
}                       

/*------------------------------------------------------------*/ 
void AzpG_Cfg::deriv(bool is_real, /* true if data is real */
                     AzPmatVar &mv, /* input: discriminator output, output: derivatives */
                     AzpG_Ev_ &evarr) {   
  evarr.add((is_real) ? AzpG_Ev_Rraw : AzpG_Ev_Fraw, mv);  
  /* loss: ln(1+exp(-yp)), derivative: -y exp(-yp)/(1+exp(-yp)) */
  double yy = (is_real) ? 1 : -1;      
  logi(yy, mv); 
}

/*------------------------------------------------------------*/ 
void AzpG_Cfg::logi(double yy, AzPmat &mo) const {
  /* loss: ln(1+exp(-yp)), derivative: -y exp(-yp)/(1+exp(-yp)) */  
  mo.to_binlogi_deriv(yy);  
}

/*------------------------------------------------------------*/ 
bool AzpG_Cfg::check_img_range(const AzPmatVar &mv_img, const char *msg, bool dont_throw) const {
  bool is_in = true; 
  const AzPmat *m = mv_img.data(); 
  if (do_pm1) { if (m->min() < -1 || m->max() > 1) is_in = false; }
  if (dont_throw) return is_in;   
  AzX::throw_if(!is_in, "AzpG_Cfg::check_img_range", msg); 
  return true; 
}

/*------------------------------------------------------------*/
/* just to make sure if tanh is done for do_pm1 */
bool AzpG_Cfg::check_fake_img_range(AzpReNet *g_net, const AzpG_Pool &pltst, 
                                    const char *msg, bool dont_throw) const {
  AzX::throw_if_null(g_net, "AzpG_Cfg::check_fake_img_range", "g_net"); 
  int d_num = MIN(minib, pltst.dataNum()); 
  AzIntArr ia; ia.range(0, d_num); 
  AzPmatVar mv_z; 
  pltst.input(ia, mv_z); 
  bool is_test = true; 
  AzPmatVar mv_fake; 
  apply_G0(g_net, is_test, mv_z, mv_fake); 
  return check_img_range(mv_fake, msg, dont_throw); 
}

/*------------------------------------------------------------*/ 
#define rand_pfx "_z_"
void AzpG_Cfg::init_rand(AzParam &azp, AzpDataRand &z_trn, AzpDataRand &z_tst, AzpDataRand &z_gen, 
                         const AzpG_dg_info *infp) const { 
  if (infp == NULL) {                           
    /*---  cold-start  ---*/
    z_trn.reset(out, azp, rand_pfx, trn_seed); 
    z_tst.reset(out, azp, rand_pfx, tst_seed); 
    z_gen.reset(out, azp, rand_pfx, gen_seed); 
  }
  else {
    /*---  warm-start  ---*/
    z_trn.copy_params_from(infp->z_data); 
    z_tst.copy_params_from(infp->z_data);     
    z_gen.copy_params_from(infp->z_data);     
    z_trn.reset_seed(trn_seed); 
    z_tst.reset_seed(tst_seed); 
    z_gen.reset_seed(gen_seed); 
  }
}  

/*------------------------------------------------------------*/ 
/* copying a neural net by writing and reading a virtual file */
void AzpG_Cfg::copy_net(AzpReNet *out_net, 
                        const AzpReNet *inp_net, 
                        AzParam &azp, const AzpData_tmpl_ *trn,
                        bool do_show, 
                        bool for_testonly) const {  
  AzX::throw_if_null(inp_net, out_net, "AzpG_Cfg::copy_net"); 
  out_net->reset(); 
  AzFileV vf("w"); /* open a virtual file for writing */
  inp_net->write(&vf); 
  vf.close(); 
  vf.open("r"); /* open the virtual file for reading */
  out_net->read(&vf); 
  vf.close(); 
  if (!do_show) out_net->deactivate_out(); 
  if (for_testonly) out_net->init_test(azp, trn); 
  else              out_net->init(azp, trn);   
  if (!do_show) out_net->activate_out(); 
}     

/*------------------------------------------------------------*/ 
/* return: do_exit */
/* check loss during the approximator update */
bool AzpG_Cfg::chk_loss(AzpReNet *net, AzBytArr &s_loss, 
               int ite, double &loss, double &count, double &last_loss_avg, 
               int &max_reduce, int &reduce_count, double &ccoeff) const {              
  double loss_avg = loss/count; 
  loss = count = 0;         
  if (last_loss_avg >= 0 && loss_avg > last_loss_avg) {
    if (reduce_count >= max_reduce) {
      AzPrint::writeln(less_out, "... Stopping as loss is not going down ... ");  
      s_loss << loss_avg << ",i," << ite+1; 
      return true;  
    }
    ccoeff *= approx_decay; 
    AzPrint::writeln(less_out, "... Reducing the step size to s0 times ", ccoeff);     
    net->multiply_to_stepsize(ccoeff);
    ++reduce_count;           
  }
  last_loss_avg = loss_avg; 
  return false; 
}

/*------------------------------------------------------------*/ 
/* testing */
void AzpG_Cfg::cfg_test(const AzpG_stat &st, 
                         AzpG_ddg &ddg, int tt, 
                         const AzpG_Ev &ev,  
                         AzpG_Pool &pltst, 
                         AzpG_ClasEval_ &cevs) const {
  AzBytArr s("clk,"); s << st.clk() << "," << st.tim(); 
  if (do_verbose) s << ",approx," << st.apprx_clk << "," << st.apprx_tim; 
  s << ",t," << tt << ",d_upd," << st.d_upd << ",g_upd," << st.g_upd;  
    
  eval_gen(s, ddg, tt, cevs, pltst);
  format_eval(s, ev); 
  AzPrint::writeln(eval_out, s);         
  AzTimeLog::print(s, out); 
}
 
/*------------------------------------------------------------*/ 
/* testing generated images */
void AzpG_Cfg::eval_gen(AzBytArr &s, 
                  AzpG_ddg &ddg, int tt, 
                  AzpG_ClasEval_ &cevs, 
                  AzpG_Pool &pldata) const {
  AzTimeLog::print("Testing ... ", out); 
  
  cevs.eval_init(test_num);             /* initialization for evaluation */
  bool is_g_test = true; 
  pldata.catchup(is_g_test, tt, ddg, test_minib); /* bring up all in the pool to G_{tt} */
  
  int mb = test_minib; 
  for (int dx = 0; dx < test_num; dx += mb) {
    int d_num = MIN(mb, test_num-dx);     
    AzPmatVar mv_fake; 
    AzIntArr ia_dxs; ia_dxs.range(dx, dx+d_num); /* dx,dx+1,...,dx+d_num-1 */
    pldata.output(ia_dxs, mv_fake); /* fake <- G_{tt}(x_i) for i=dx,...,dx+d_num-1 */
    gen_last_step(mv_fake);             /* truncate into [-1,1] */
    cevs.eval_clas(mv_fake, dx, d_num); /* evaluate using a classifier */
  }
  cevs.show_eval(s); 

  AzTimeLog::print("Done with testing ... ", out); 
} 

/*------------------------------------------------------------*/ 
void AzpG_Cfg::check_input(const AzpDataSet_ &r_ds, const AzpG_dg_info *infp) {
  const char *eyec = "AzpG_Cfg::check_input";
  AzX::throw_if_null(r_ds.trn_data(), AzInputError, eyec, "r_trn (data)"); 
  AzX::no_support(r_ds.trn_data()->datasetNum() != 1, eyec, "r_trn: multi"); 
  if (infp != NULL) { 
    AzX::throw_if(infp->do_pm1 != do_pm1, AzInputError, eyec, 
                  "The image scaling option has changed?!  Warm-start failed."); 
    AzX::throw_if(!infp->d_tmpl.is_same_x_tmpl(*r_ds.trn_data()), AzInputError, eyec, 
                  "The real data template has changed?!  Warm-start failed."); 
  }
}  

/*------------------------------------------------------------*/ 
void AzpG_Cfg::approximate(AzpG_ddg &ddg, int tt, AzpG_Pool &pltrn, int x_epo, double ini_ccoeff) {
  const char *eyec = "AzpG_Cfg::approximate";       
  AzBytArr s("tt="); s << tt << " ini_coeff=" << ini_ccoeff; 
  AzTimeLog::print("AzpG_Cfg::approximate begins ... ", s.c_str(), less_out);
  AzTimeLog::print("approximate: preparing data ... ", less_out); 
  pltrn.catchup(!do_gtr, tt, ddg, minib); /* make all in the pool G_{tt} */
  AzTimeLog::print("preparation is done ... ", less_out);
   
  approx(ddg.g_net, pltrn, x_epo, ini_ccoeff); 

  AzTimeLog::print("AzpG_Cfg::approximate ends ... ", less_out);   
}  
  
/*------------------------------------------------------------*/ 
/* update the approximator              */
/* goal: minimize |G_T(z) - G_0(z)|^2/2 */
/* NOTE: Approximator tilde{G} serves as the initial generator G_0 in ICFG. */
void AzpG_Cfg::approx(AzpReNet *g_net, AzpG_xy_ &pltrn, int x_epo, double ini_ccoeff) {  
  int max_reduce = approx_redmax, reduce_count = 0;  
  double ccoeff = ini_ccoeff; 
  g_net->multiply_to_stepsize(ccoeff, &less_out); /* reset the learning rate */

  AzBytArr s_loss("gloss,"); 
  double last_loss_avg = -1; 
  for (int epo = 0; epo < x_epo; ++epo) { /* epochs */
    int data_num = pltrn.dataNum(); 
    AzIntArr ia_dxs; ia_dxs.range(0, data_num); 
    AzTools::shuffle2(ia_dxs, &rg); /* randomize the order of data indexes */
    double loss = 0, count=0; 
    for (int ix = 0; ix < data_num; ix += minib) {
      int d_num = MIN(minib, data_num-ix); 
      
      AzIntArr my_ia_dxs(ia_dxs.point()+ix, d_num); 
      AzPmatVar mv_z, /* z: random vectors */ mv_y; /* G_T(z) */
      pltrn.input_output(my_ia_dxs, mv_z, mv_y); 

      bool is_g_test = false;      
      AzPmatVar mv_p; 
      apply_G0(g_net, is_g_test, mv_z, mv_p); /* p <- G_0(z) */         
      mv_p.data_u()->sub(mv_y.data());        /* p -= y */
      loss += mv_p.data()->squareSum()/2; /* loss += sum_i (p_i-y_i)^2/2 */
      count += mv_p.dataNum(); 
      g_net->down(mv_p); /*bprop */
      g_net->flush();        
    }
    if (epo == 0 || epo == x_epo-1) s_loss << loss/count << ",";     
    AzBytArr s("   epo,"); s << epo+1 << "," << count << ",loss,"; 
    s << loss/count; 
    AzTimeLog::print(s.c_str(), less_out); /* show loss */

    bool do_exit = chk_loss(g_net, s_loss, epo, loss, count, last_loss_avg, 
                            max_reduce, reduce_count, ccoeff); 
    if (do_exit) break; 
  } /* epochs */

  g_net->multiply_to_stepsize(ini_ccoeff, &less_out); /* reset the learning rate */
  AzPrint::writeln(less_out, s_loss);         
}

/*------------------------------------------------------------*/ 
/* apply the random data generator to z (gaussian vectors)    */
/* used for initializing the approximator                     */
void AzpG_Cfg::gen_rand(int d_num, int cc, int sz, 
                        const AzpDataRand &z_trn,
                        const AzPmat &m_w, /* random data generator */
                        AzPmatVar &mvo, /* output */
                        AzPmatVar *mv_z) /* optional output */ const {
  const char *eyec = "AzpG_Cfg::gen_rand";           
  AzPmatVar mv; z_trn.gen_data(d_num, mv); 
  if (mv_z != NULL) mv_z->set(&mv); 
  
  AzPmat md; md.prod(&m_w, mv.data(), true, false); /* md <- w^T mv */
  md.change_dim(cc, md.size()/cc); 
  AzX::throw_if(md.colNum() != sz*d_num, eyec, "shape mismatch"); 
  AzIntArr ia; 
  for (int dx = 0; dx < d_num; ++dx) { ia.put(dx*sz); ia.put((dx+1)*sz); }
  mvo.set(&md, &ia); 
  gen_last_step(mvo); 
}

/*------------------------------------------------------------*/ 
int AzpG_Cfg::get_size(const AzpData_tmpl_ &d_tmpl) const {
  const char *eyec = "AzpG_Cfg::get_size"; 
  int shape_dim = d_tmpl.dimensionality(); 
  AzX::no_support(shape_dim != 2, eyec, "Anything other than 2D data"); 
  AzX::no_support(d_tmpl.size(0)<=0 || d_tmpl.size(1)<=0, eyec, "Variable-sized data"); 
  int sz = d_tmpl.size(0)*d_tmpl.size(1); /* wid*hei */
  AzPrint::writeln(out, "AzpG_Cfg::get_size,size=", sz); 
  return sz; 
}

/*-------------------------------------------------------------*/ 
/* generate a random data generator and train the approximator */
/* to approximate its behavior                                 */
 void AzpG_Cfg::random_init(AzpReNet *g_net, 
                            const AzpData_tmpl_ &d_tmpl, /* to get size */
                            const AzpDataRand &z_trn) {
  const char *eyec = "AzpG_Cfg::random_init";        
  AzXi::throw_if_nonpositive(rd_wini, eyec, kw_rd_wini); 
  AzX::throw_if_null(g_net, eyec, "approximator"); 
  check_rg(eyec); 
  
  int rd_trn_num = poolsz; 
  int rd_epo = approx_epo; 
  int cc = d_tmpl.xdim(), sz = get_size(d_tmpl); 
  
  /*---  make a random data generator of one linear layer  ---*/
  AzDmat md(z_trn.xdim(), sz*cc); rg.gaussian(rd_wini, &md); 
  AzPmat m_w(&md);  /* this is the weights of the linear layer */
  
  /*---  produce training data for training the approximator  ---*/   
  AzpG_xy0 xy;  
  for (int dx = 0; dx+minib <= rd_trn_num; dx += minib) {
    AzPmatVar mv_g, mv_z; 
    gen_rand(minib, cc, sz, z_trn, m_w, mv_g, &mv_z); 
    if (dx == 0) xy.init(mv_z, mv_g, rd_trn_num); 
    xy.put(dx, mv_z, mv_g); 
  }
     
  /*---  train the approximator  ---*/
  approx(g_net, xy, rd_epo);
}

/*------------------------------------------------------------*/ 
int AzpG_Cfg::g_config(AzpG_ddg &ddg, AzParam &g_azp, 
                          const AzpData_tmpl_ &z_trn, const AzpData_tmpl_ &r_trn) {   
  int g_outdim = ddg.g_net->init(g_azp, &z_trn); 
  if (do_verbose) ddg.g_net->show_layer_stat("approximator:", true); 
  AzX::throw_if(g_outdim != r_trn.xdim(), "AzpG_Cfg::g_config", 
                "Dim mismatch: output of generator vs. input of discriminator");   
  return g_outdim; 
}  

/*------------------------------------------------------------*/ 
void AzpG_Cfg::d_g_init(AzpReNet *d_net, int ini_tt, AzpG_ddg &ddg, AzParam &g_azp, 
                      const AzpDataRand &z_trn, const AzpData_ &r_trn, AzpG_stat &st, 
                      AzpG_Pool &pltst, AzpG_ClasEval_ &cevs) {
  if (ini_tt > 0) return;                        
  /*---  initialize the approximator (g_net)  ---*/
  AzTimeLog::print("Initiazing the approximator ... ", out); 
  AzClock myclk(true);    
  random_init(ddg.g_net, r_trn, z_trn); 
  myclk.tick(st.apprx_clk, st.apprx_tim);    
}
  
/*------------------------------------------------------------*/ 
void AzpG_Cfg::d_config(AzpReNet *d_net, AzpG_ddg &ddg, AzParam &d_azp, 
                          int ini_tt, const AzpData_tmpl_ &r_trn) { 
  if (ini_tt > 0) { /* warm-start from a saved model */
    bool do_show = true; 
    AzPrint::writeln(out, "Copying the initial d_net ... "); 
    copy_net(d_net, ddg.d_net(ini_tt-1), d_azp, &r_trn, do_show);
    for (int tt = 0; tt < ini_tt; ++tt) ddg.d_net(tt)->init_noshow(d_azp, &r_trn); 
    for (int tt = ini_tt; tt < ddg.dNum(); ++tt) ddg.set_eta(tt, cfg_eta); 
  }
  else { /* cold-start */
    d_net->init(d_azp, &r_trn);
    ddg.set_eta(cfg_eta); 
  }
  if (do_verbose) d_net->show_layer_stat("discriminator:", true); 
} 

/*------------------------------------------------------------*/ 
bool AzpG_Cfg::check_progress(int tt, const AzpG_stat &st, AzpG_Ev &ev) const {
  bool do_exit = false;
  double clk = st.clk();
  double ddiff = ev.avg(AzpG_Ev_Rraw) - ev.avg(AzpG_Ev_Fraw);
  if (inc<0 || inc>0 && st.g_upd%inc == 0) {
    AzBytArr s("g_upd="); s<<st.g_upd<<" clk="<<clk<<" Dreal-Dgen="<<ddiff;
    s << " c," << ev.get_count(AzpG_Ev_Fraw);
    AzTimeLog::print(s.c_str(), out); /* show progress */
  }
  if (cfg_diff_max > 0 && ddiff > cfg_diff_max && st.g_upd > tt && clk > clk_min) {
    show_eval(ev);
    AzTimeLog::print("Dreal-Dgen seems to be exploding ... exiting", out);
    do_exit = true; /* exit as it's failing */
  }
  ev.reset();
  return do_exit; 
}
      
/*------------------------------------------------------------*/ 
void AzpG_Cfg::update_G(AzpG_ddg &ddg, int tt, const AzpReNet *d_net, AzParam &d_azp, 
                        const AzpData_ &r_trn) {
  copy_net(ddg.d_net(tt), d_net, d_azp, &r_trn); 
}

/*------------------------------------------------------------*/ 
int AzpG_Cfg::loop_d_update(int my_U, AzpReNet *d_net, 
                const AzpData_ *r_trn, const int *dxs, int ini_index,
                AzpG_Pool &pltrn, AzpG_ddg &ddg, int tt,
                AzpG_stat &st, AzpG_Ev_ &evarr) {                
  /*---  update D  ---*/
  int index = ini_index; 
  for (int dupd = 0; dupd < my_U; ++dupd, index += minib) {
    AzClock myclk(true); /* beSilent=true */
    d_update(d_net, r_trn, dxs, index, pltrn, ddg, tt, evarr);     
    myclk.tick(st.tune_clk, st.tune_tim);    
    ++st.d_upd;    
  } /* dupd */
  d_net->end_of_epoch();   
  return index; 
}

/*------------------------------------------------------------*/ 
void AzpG_Cfg::d_update(AzpReNet *d_net, 
                const AzpData_ *r_trn, const int *dxs, int index,
                AzpG_Pool &pltrn, AzpG_ddg &ddg, int tt,                 
                AzpG_Ev_ &evarr) {           
  bool is_test = false;       
  /*---  fprop and bprop with real data  ---*/
  AzPmatVar mv; 
  real_up(is_test, d_net, r_trn, dxs+index, minib, mv, evarr);
  d_net->down(mv); /* bprop */

  /*---  fprop and bprop with fake data  ---*/
  AzPmatVar mv_fake; 
  pltrn.pick_data(!do_gtr, tt, ddg, minib, mv_fake, minib); 
  gen_last_step(mv_fake, true); 
  d_net->up(is_test, mv_fake, mv);  
  deriv_fake(mv, evarr);        
  d_net->down(mv); /* bprop */
  
  d_net->flush();  /* update D's weights */ 
}

/*------------------------------------------------------------*/ 
void AzpG_Cfg::save_ddg(const AzpG_dg_info &info, 
                           int tt, double clk, AzpG_ddg &ddg) {
  if (s_save_fn.length() <= 0) return; 
  ddg.set_num_in_file(tt); 
  AzBytArr s_clk; clk_str(clk, s_clk);  
  ddg.write_nets(s_save_fn, info, s_clk, out);
} 

/*------------------------------------------------------------*/ 
/* generate images; called during training                    */
void AzpG_Cfg::gen_in_trn(AzpG_ddg &ddg, const AzpData_tmpl_ &d_tmpl, 
                             AzpDataRand &z_gen, 
                             int tt, double clk) const {
  if (gen_num <= 0 || s_gen_fn.length() <= 0) return; 
  AzTimeLog::print("Generating images ... ", out); 
  if (do_reset_gen_seed) z_gen.reset_seed(gen_seed); 
  AzBytArr s_clk; clk_str(clk, s_clk); 
  generate_ppm(ddg, d_tmpl, z_gen, tt, s_clk.c_str()); 
}                               

/*------------------------------------------------------------*/ 
/* input: gen_num, s_gen_fn, do_no_collage */
/* output: ppm/pgm files    */
void AzpG_Cfg::generate_ppm(AzpG_ddg &ddg, const AzpData_tmpl_ &d_tmpl, 
                            const AzpDataRand &z_gen, 
                            int tt, const char *clk_str) const {
  int width = d_tmpl.size(0), height = d_tmpl.size(1), cc = d_tmpl.xdim(); 
  AzBytArr s_fn; s_fn << s_gen_fn; 
  if (clk_str != NULL) s_fn << clk_str;   
  AzpG_unsorted_images un(s_fn, gen_num, do_pm1, width, height, cc, do_no_collage); 
  int mb = test_minib; 
  bool do_ms = (gen_num >= 1000); 
  int mlinc = gen_num / 50, milestone = mlinc;     
  for (int dx = 0; dx < gen_num; dx += mb) {
    if (do_ms) AzTools::check_milestone(out, milestone, dx, mlinc);     
    int d_num = MIN(mb, gen_num-dx); 
    AzPmatVar mv_gen; gen(tt, z_gen, d_num, ddg, mv_gen);      
    un.proc(mv_gen);   
  }
  if (do_ms) AzTools::finish_milestone(out, milestone); 
  un.flush(); 
}

/*------------------------------------------------------------*/ 
void AzpG_Cfg::gen(int tt, const AzpDataRand &z_data, int d_num, 
                      AzpG_ddg &ddg, 
                      AzPmatVar &mv_fake) const { /* output */
  AzPmatVar mv_z; 
  z_data.gen_data(d_num, mv_z); 
  bool is_g_test = true; 
  apply_G0(ddg.g_net, is_g_test, mv_z, mv_fake);
  for (int tx = 0; tx < tt; ++tx) apply_gt(mv_fake, ddg.d_net(tx), ddg.eta(tx), mv_z); 
  gen_last_step(mv_fake); 
}

/*------------------------------------------------------------*/  
/*------------------------------------------------------------*/ 
void AzpG_Cfg::generate(AzpG_ddg &ddg, const AzpG_dg_info &info, 
                           AzParam &g_azp, AzParam &d_azp, AzParam &azp) {
  const char *eyec = "AzpG_Cfg::generate";

  do_pm1 = info.do_pm1;
  if (info.minib > 0) test_minib=minib=info.minib; 
  else                test_minib=minib=64; 
  ddg.check(); 

  #define kw_gen_t "cfg_t="
  #define kw_do_genbin "Bin"
  int gen_t=-1, gen_seed=1; 
  bool do_genbin=false; 
  /*-----------------------------------------*/ 
  AzPrint o(out); 
  azp.vInt(o, kw_minib, minib);  test_minib = minib; 
  azp.vInt(o, kw_gen_t, gen_t); 
  azp.vInt(o, kw_gen_seed, gen_seed);    
  azp.vStr(o, kw_gen_fn, s_gen_fn); 
  azp.vInt(o, kw_gen_num, gen_num); 
  azp.swOn(o, do_genbin, kw_do_genbin); 
  if (!do_genbin) azp.swOn(o, do_no_collage, kw_do_no_collage); 
  azp.swOn(o, do_verbose, kw_do_verbose); 
  o.printEnd(); 
  AzXi::throw_if_nonpositive(minib, eyec, kw_minib); 
  AzXi::throw_if_nonpositive(gen_num, eyec, kw_gen_num); 
  AzXi::throw_if_empty(s_gen_fn, eyec, kw_gen_fn); 
  azp.check(out);  
  /*-----------------------------------------*/
  
  if (do_verbose) AzPrint::writeln(out, "=====  g_net  =====");   
  AzpDataRand z_gen; 
  z_gen.copy_params_from(info.z_data); 
  z_gen.reset_seed(gen_seed);  
  if (!do_verbose) ddg.g_net->deactivate_out(); 
  int g_outdim = ddg.g_net->init_test(g_azp, &z_gen); 

  if (do_verbose) AzPrint::writeln(out, "=====  d_net  ====="); 
  int ini_tt = MAX(0, ddg.get_num_in_file()); 
  for (int tt = 0; tt < ini_tt; ++tt) {
    if (tt != 0 || !do_verbose) ddg.d_net(tt)->deactivate_out(); 
    ddg.d_net(tt)->init_test(d_azp, &info.d_tmpl); 
    ddg.d_net(tt)->activate_out(); 
  }

  AzPrint::writeln(out, "T in file = ", ini_tt); 
  int tt = ini_tt; 
  if (gen_t >= 0) {
    AzX::throw_if(gen_t > ini_tt, AzInputError, eyec, kw_gen_t, " is too large");  
    tt = gen_t; 
  }
  AzPrint::writeln(out, "t=", tt); 
  
  if (do_genbin) generate_bin(ddg, info.d_tmpl, z_gen, tt); 
  else           generate_ppm(ddg, info.d_tmpl, z_gen, tt); 
}

/*------------------------------------------------------------*/ 
/* input: gen_num, s_gen_fn */
/* output: generated images in the xbin format */
void AzpG_Cfg::generate_bin(AzpG_ddg &ddg, const AzpData_tmpl_ &d_tmpl, 
                            const AzpDataRand &z_gen, 
                            int tt) const {
  int width = d_tmpl.size(0), height = d_tmpl.size(1), cc = d_tmpl.xdim(); 
  int mb = test_minib;  

  bool do_ms = (gen_num >= 1000); 
  
  AzBytArr s_fn(s_gen_fn.c_str(), ".xbin"); 
  AzFile ofile(s_fn.c_str());   
  ofile.open("wb"); 
  ofile.writeInt(cc);
  ofile.writeInt(width);
  ofile.writeInt(height); 
  ofile.writeInt(gen_num); 
  int mlinc = gen_num / 50, milestone = mlinc;   
  for (int dx = 0; dx < gen_num; dx += mb) {
    if (do_ms) AzTools::check_milestone(out, milestone, dx, mlinc);     
    int d_num = MIN(mb, gen_num-dx); 
    AzPmatVar mv_gen;
    gen(tt, z_gen, d_num, ddg, mv_gen);   
    AzpG_Tools::to_img_bin(mv_gen, width, height, cc, do_pm1, &ofile);       
  }  
  ofile.close(true);
  if (do_ms) AzTools::finish_milestone(out, milestone);   
}