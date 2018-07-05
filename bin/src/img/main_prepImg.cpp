#define _AZ_MAIN_
#include "AzUtil.hpp"
#include "AzStrPool.hpp"
#include "AzTools.hpp"

void help() {
  cout << "action: mnist_to_xbin | mnist_to_y | ppmpgm_to_xbin | xbin_to_ppmpgm" << endl;
}

/*----------------------------------------------------------------*/
void write_pixels(AzFile &ofile, 
                  const AzByte *img, 
                  int wid, int hei, int cc, 
                  int padsz) {
  int len = wid*hei*cc;                     
  if (padsz == 0) ofile.writeBytes(img, len); 
  else {
    AzBytArr byt_pad; 
    const AzByte *pads = byt_pad.reset(cc, 0); 
    int ipos = 0; 
    for (int hx = -padsz; hx < hei+padsz; ++hx) {
      for (int wx = -padsz; wx < wid+padsz; ++wx) {
        if (wx < 0 || wx >= wid || hx < 0 || hx >= hei) {
          ofile.writeBytes(pads, cc); 
        }
        else {
          ofile.writeBytes(img+ipos, cc); 
          ipos += cc; 
        }
      }
    }
  } 
}    

/*----------------------------------------------------------------*/
int parse_ppmpgm_header(const AzByte *data, int len, /* input */
                        int &width, int &height, int &maxval, int &cc, /* output */ 
                        const char *fn) {
  const char *eyec = "parse_ppmpgm_header";                
  const AzByte *wp0 = data, *wp1 = NULL; 
  AzBytArr s_magic; 
  for (int ix = 0; ix < 4; ++ix) {    
    for (           ; wp0 < data+len; ++wp0) if (*wp0 >  0x20) break; 
    for (wp1 = wp0+1; wp1 < data+len; ++wp1) if (*wp1 <= 0x20) break; 
    AzX::throw_if(wp0 >= data+len, AzInputError, eyec, "4 components were expected: ", fn); 

    if      (ix == 0) s_magic.reset(wp0, (int)(wp1-wp0)); 
    else if (ix == 1) width  = atol((char *)wp0); 
    else if (ix == 2) height = atol((char *)wp0); 
    else if (ix == 3) maxval = atol((char *)wp0); 
    wp0 = wp1+1; 
  }
  if      (s_magic.equals("P6")) cc = 3; 
  else if (s_magic.equals("P5")) cc = 1; 
  else {
    AzX::throw_if(true, AzInputError, eyec, 
                  "P5 (pgm) or P6 (ppm) was expected at the beginning of the file: ", fn);  
  } 
  return (int)(wp0-data); 
}
    
/*----------------------------------------------------------------*/
void ppmpgm_to_xbin(int argc, const char *argv[]) {
  const char *eyec = "ppmpgm_to_xbin"; 
  if (argc < 3) {
    cout << eyec << ": out_fn padsz fn0 [ fn1 [ fn2 ... ] ]" << endl; 
    return;     
  }
  int argx = 0; 
  const char *out_fn = argv[argx++]; 
  int padsz = atol(argv[argx++]); 
  int f_num = argc - argx; 
  
  cout << "output file: " << out_fn << ", padding: " << padsz; 
  cout << ", # of input files: " << f_num << endl; 
  
  int width = 0, height = 0, cc = 0; 
  AzFile ofile(out_fn); ofile.open("wb"); 
  const char **fns = argv+argx; 
  for (int fx = 0; fx < f_num; ++fx) {
    /*---  read one ppm/pgm file  ---*/
    AzFile file(fns[fx]); file.open("rb"); 
    int sz = (int)file.size();
    AzBytArr byt_data; 
    AzByte *data = byt_data.reset(sz+256, 0); 
    file.readBytes(data, sz); 
    file.close(); 
    
    /*---  parse the pgm/ppm header  ---*/
    int wid, hei, maxval, my_cc; 
    int hdr_len = parse_ppmpgm_header(data, sz, wid, hei, maxval, my_cc, fns[fx]); 
    AzX::throw_if(maxval >= 256, AzInputError, eyec, "Expected maxval<256: ", fns[fx]);   
  
    /*****  write xbin  *****/
    if (fx == 0) { /* first file */
      width = wid; height = hei; cc = my_cc; 
      cout << "width: " << width << " height: " << height << endl; 
      /*---  write the header if this is the first file  ---*/
      ofile.writeInt(cc); 
      ofile.writeInt(width+2*padsz); 
      ofile.writeInt(height+2*padsz); 
      ofile.writeInt(f_num); 
    }
    else { 
      AzX::throw_if(wid != width, AzInputError, eyec, "width must be fixed.");
      AzX::throw_if(hei != height, AzInputError, eyec, "height must be fixed.");
      AzX::throw_if(my_cc != cc, AzInputError, eyec, "pgm/ppm cannot be mixed."); 
    }
    
    /*---  write pixels  ---*/
    int img_len = cc * width * height; 
    AzX::throw_if(sz < hdr_len+img_len, AzInputError, eyec, "The file is too small? ", fns[fx]);   
    write_pixels(ofile, data+sz-img_len, width, height, cc, padsz);      
  }
  ofile.close(true); 
}

/*----------------------------------------------------------------*/
void xbin_to_ppmpgm(int argc, const char *argv[]) {
  const char *eyec = "xbin_to_ppmpgm"; 
  if (argc != 2) {
    cout << eyec << ": inp_fn out_fn_stem" << endl; 
    return;     
  }
  int argx = 0; 
  const char *inp_fn = argv[argx++]; 
  const char *ofn_stem = argv[argx++]; 

  AzFile ifile(inp_fn); 
  ifile.open("rb"); 
  int cc = ifile.readInt(); 
  int width = ifile.readInt(); 
  int height = ifile.readInt(); 
  int f_num = ifile.readInt(); 
  int len = width*height*cc; 

  cout << "input file: " << inp_fn; 
  cout << ", output file: " << ofn_stem << "*" << endl; 
  cout << width << " x " << height << " x " << cc << endl; 
  cout << "# of files: " << f_num << endl; 
  
  AzBytArr byt_buff; 
  AzByte *buff = byt_buff.reset(len+256, 0); 
  
  const char *ofn_ext = (cc == 1) ? ".pgm" : ".ppm"; 
  const char *magic   = (cc == 1) ? "P5"   : "P6"; 
  
  AzTimeLog::print("Generating ppm/pgm files ... ", log_out); 
  for (int fx = 0; fx < f_num; ++fx) {   
    /*---  write one ppm/pgm file  ---*/
    AzBytArr s_fn(ofn_stem); s_fn << fx << ofn_ext; 
    AzFile ofile(s_fn.c_str()); 
    ofile.open("wb"); 
    
    AzBytArr s_head(magic, " "); 
    s_head << width << " " << height << " 255"; 
    s_head.nl();     
    s_head.writeText(&ofile); 
    
    ifile.readBytes(buff, len); 
    ofile.writeBytes(buff, len); 
    ofile.close();     
  }
  ifile.close(); 
  AzTimeLog::print("Done ... ", log_out); 
}

/*----------------------------------------------------------------*/
void mnist_to_xbin(int argc, const char *argv[]) {
  const char *eyec = "mnist_to_xbin";   
  if (argc != 2 && argc != 3) {
    cout << eyec << ": inp_fn out_fn [padsz]" << endl; 
    return; 
  }
  int argx = 0; 
  const char *inp_fn = argv[argx++]; /* {train|t10k}-images-idx3-ubyte */
  const char *out_fn = argv[argx++]; 
  int padsz = 2; 
  if (argx < argc) padsz = atol(argv[argx++]); 
  AzByte pad = 0; 
  
  AzFile ifile(inp_fn); ifile.open("rb"); 
  AzFile ofile(out_fn); ofile.open("wb"); 
  int isz = (int)ifile.size(); 
  for (int ix = 0; ix < 4; ++ix) int dummy = ifile.readInt(); 
  int wid = 28, hei = 28; 
  AzX::throw_if((isz-16)%(wid*hei) != 0, eyec, "file size is wrong ... "); 
  int num = (isz-16)/(wid*hei); 
  cout << eyec << ": #examples=" << num << " width=" << wid << " height=" << hei; 
  cout << " padding=" << padsz << endl; 
  int cc = 1; 
 
  ofile.writeInt(cc); 
  ofile.writeInt(wid+2*padsz); 
  ofile.writeInt(hei+2*padsz); 
  ofile.writeInt(num);   
  
  int sz = wid*hei*cc; 
  AzBytArr byt_data, byt_padded; 
  AzByte *data = byt_data.reset(sz, 0); 
  AzByte *padded = byt_padded.reset((wid+2*padsz)*(hei+2*padsz)*cc, 0); 
  for (int dx = 0; dx < num; ++dx) {
    ifile.readBytes(data, sz); 
    write_pixels(ofile, data, wid, hei, cc, padsz);  
  }
  AzX::throw_if(ifile.tell() != ifile.size(), eyec, "more data?!");   
  ifile.close(); 
  ofile.close(true);   
}  

/*----------------------------------------------------------------*/
void mnist_to_y(int argc, const char *argv[]) {
  const char *eyec = "mnist_to_y"; 
  if (argc != 2) {
    cout << eyec << ": inp_fn out_fn" << endl; 
    return; 
  }
  int argx = 0; 
  const char *inp_fn = argv[argx++]; /* {train|t10k}-labels-idx1-ubyte */
  const char *out_fn = argv[argx++]; 
  
  AzFile ifile(inp_fn); ifile.open("rb"); 
  int isz = (int)ifile.size(); 
  int num = isz - 8; 
  cout << eyec << ": #examples=" << num << endl; 
  for (int ix = 0; ix < 2; ++ix) int dummy = ifile.readInt(); 
  AzBytArr s_data; 
  AzByte *data = s_data.reset(num, 0); 
  ifile.readBytes(data, num); 
  ifile.close(); 
  
  AzFile ofile(out_fn); ofile.open("wb");   
  AzBytArr s("sparse 10"); s.nl(); 
  s.writeText(&ofile); 
  for (int dx = 0; dx < num; ++dx) {
    int lab = (int)data[dx]; 
    AzX::throw_if(lab < 0 || lab >= 10, eyec, "label is out of range."); 
    s.reset(); s << lab; s.nl(); 
    s.writeText(&ofile); 
  }
  ofile.close(true); 
}  

/*******************************************************************/
/*     main                                                        */
/*******************************************************************/
int main(int argc, const char *argv[]) 
{
  AzException *stat = NULL; 

  if (argc < 2) {
    help(); 
    return -1; 
  }

  const char *action = argv[1]; 

  try {
    Az_check_system2_(); 
    
    if      (strcmp(action, "mnist_to_xbin")  == 0) mnist_to_xbin (argc-2, argv+2);     
    else if (strcmp(action, "mnist_to_y")     == 0) mnist_to_y    (argc-2, argv+2);  
    else if (strcmp(action, "ppmpgm_to_xbin") == 0) ppmpgm_to_xbin(argc-2, argv+2); 
    else if (strcmp(action, "xbin_to_ppmpgm") == 0) xbin_to_ppmpgm(argc-2, argv+2); 
    else {
      help(); 
      return -1; 
    }
  }
  catch (AzException *e) {
    stat = e; 
  }

  if (stat != NULL) {
    cout << stat->getMessage() << endl; 
    return -1; 
  }

  return 0; 
}

