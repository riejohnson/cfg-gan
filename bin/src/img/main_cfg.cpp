#define _AZ_MAIN_
#include "AzUtil.hpp"
#include "AzpMain_Cfg.hpp"
#include "AzpG_Cfg.hpp"
#include "AzpG_ClasEval.hpp"

extern AzPdevice dev; 

/*-----------------------------------------------------------------*/
void help() {
#ifdef __AZ_GPU__  
  cout << "Arguments:  gpu#[:mem]  action  parameters" <<endl; 
#else
  cout << "Arguments:  _  action  parameters" <<endl; 
#endif 
  cout << "   action: cfg_train | cfg_gen | eval_img"<<endl; 
  cout << endl; 
  cout << "      cfg_train: to train CFG-GAN." << endl; 
  cout << "      cfg_gen:   to generate images using a trained CFG-GAN model." << endl; 
  cout << "      eval_img:  to evaluate images using a classifier." << endl;   
}

/*******************************************************************/
/*     main                                                        */
/*******************************************************************/
int main(int argc, const char *argv[]) {
  AzException *stat = NULL; 
  
  if (argc < 3) {
    help(); 
    return -1; 
  }
  const char *gpu_param = argv[1]; 
  int ix; 
  for (ix = 2; ix < argc; ++ix) {
    argv[ix-1] = argv[ix]; 
  }
  --argc; 

  int gpu_dev = dev.setDevice(gpu_param); 
  if (gpu_dev < 0) {
    AzPrint::writeln(log_out, "Using CPU ... "); 
  }
  else {
    AzPrint::writeln(log_out, "Using GPU#", gpu_dev); 
  }  

  const char *action = argv[1]; 
  AzBytArr s_action(action); 
  int ret = 0; 
  try {
    Az_check_system2_(); 

    AzpG_Cfg cfg; 
    AzpG_ClasEvalArr cevs; 
    AzpMain_Cfg g;        
    if      (s_action.equals("cfg_train"  )) g.cfg_train(cfg, cevs, argc-2, argv+2, s_action);      
    else if (s_action.equals("cfg_gen"    )) g.cfg_gen(cfg, argc-2, argv+2, s_action);      
    else if (s_action.equals("eval_img"   )) g.eval_img(cevs, argc-2, argv+2, s_action);     
    else if (s_action.equals("gen_ppm"    )) g.gen_ppm(argc-2, argv+2, s_action);     
    
    else if (s_action.equals("class_train"  )) g.renet(argc-2, argv+2, s_action); 
    else if (s_action.equals("class_predict")) g.predict(argc-2, argv+2, s_action);      
    else {
      help(); 
      ret = -1; 
      goto labend; 
    }
  } 
  catch (AzException *e) {
    stat = e; 
  }
  if (stat != NULL) {
    cout << stat->getMessage() << endl; 
    ret = -1; 
    goto labend;  
  }

labend: 
  dev.closeDevice();   
  return ret; 
}
