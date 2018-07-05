/* * * * *
 *  AzpG_Tools.cpp
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
#include "AzpG_Tools.hpp"

/*------------------------------------------------------------*/ 
/* static */
void AzpG_Tools::to_img_bin(const AzPmatVar &mv, 
                           int width, int height, int cc, 
                           bool do_pm1, 
                           AzFile *ofile) {
  const char *eyec = "AzpReApp_::to_img_bin"; 
  if (mv.dataNum() <= 0) return; 
  AzX::throw_if_null(ofile, eyec, "ofile"); 
  AzDmat md; to_img(mv, md, width*height*cc, do_pm1); 
  int d_num = md.colNum(); 
  for (int dx = 0; dx < d_num; ++dx) {
    AzBytArr s_pix;     
    for (int row = 0; row < md.rowNum(); ++row) s_pix.concat((AzByte)md.get(row,dx)); 
    s_pix.writeText(ofile); /* writing bytes, not text */ 
  }
} 

/*------------------------------------------------------------*/ 
/* static */
void AzpG_Tools::to_img(const AzPmatVar &mv, AzDmat &md, int sz, bool do_pm1) {
  const char *eyec = "AzpG_Tools::_to_img"; 
  AzPmat m(mv.data()); 
  m.change_dim(m.size()/mv.dataNum(), mv.dataNum());   
  if (do_pm1) {
    m.multiply(128);     
    m.add(128); 
  }
  m.truncate(0, 255); 
  AzX::throw_if(m.rowNum() != sz, eyec, "size mismatch");  
  m.get(&md); 
}

/*------------------------------------------------------------*/ 
/* static */
void AzpG_Tools::gen_ppm_header(int width, int height, int cc, 
                                AzBytArr &s_head, 
                                AzBytArr *s_ext) /* may be NULL */ { 
  s_head.reset();   
  if      (cc == 3) s_head << "P6 "; 
  else if (cc == 1) s_head << "P5 ";
  else AzX::throw_if(true, "AzpG_Tools::gen_ppm_header", "#channels must be 1 or 3."); 
  s_head << width << " " << height << " 255"; s_head.nl(); 
  if (s_ext != NULL) {
    if (cc == 3) s_ext->reset(".ppm"); 
    else         s_ext->reset(".pgm"); 
  }  
}                                  
  
/*------------------------------------------------------------*/ 
/* static */
void AzpG_Tools::to_img_ppm(const AzPmatVar &mv, 
                           int width, int height, int cc, 
                           bool do_pm1, 
                           const char *fn_stem, int dx_begin,
                           const AzStrPool *sp) {
  const char *eyec = "AzpG_Tools::to_img_ppm"; 
  if (mv.dataNum() <= 0) return; 
  AzX::throw_if_null(fn_stem, eyec, "fn_stem"); 
  AzX::throw_if(sp != NULL && sp->size() != mv.dataNum(), eyec, 
                "#strings given in sp must be #data"); 
  AzDmat md; to_img(mv, md, width*height*cc, do_pm1);   
  
  AzBytArr s_head, s_ext; 
  gen_ppm_header(width, height, cc, s_head, &s_ext); 
  int d_num = md.colNum(); 
  for (int dx = 0; dx < d_num; ++dx) {
    AzBytArr s_pix;     
    for (int row = 0; row < md.rowNum(); ++row) s_pix.concat((AzByte)md.get(row,dx)); 
    
    AzBytArr s_fn; 
    s_fn << fn_stem;
    if (sp != NULL) s_fn << sp->c_str(dx); 
    s_fn << "-" << dx+dx_begin << s_ext; 
    AzFile file(s_fn.c_str()); file.open("wb"); 
    s_head.writeText(&file);    
    s_pix.writeText(&file); /* writing bytes, not text */
    file.close(true); 
  }
} 

/*----------------------------------------------------------------*/
void AzpG_Tools::gen_collage_ppm(const AzpData_ *data, bool do_pm1, 
          const AzIntArr &ia_dxs, int gap, 
          const char *out_fn, int wnum, int hnum) {
  const char *eyec = "AzpG_Tools::gen_collage_ppm"; 
  AzX::throw_if_null(data, eyec, "data"); 
  int num = wnum*hnum; 
  AzX::throw_if(ia_dxs.size() != num, eyec, "numbers don't match"); 
  int w0 = data->size(0), h0 = data->size(1), cc = data->xdim(); 

  int wid = w0*wnum + gap*(wnum-1), hei = h0*hnum + gap*(hnum-1); 
  AzFile file(out_fn); file.open("wb"); 

  AzBytArr s; 
  gen_ppm_header(wid, hei, cc, s);   
  s.writeText(&file);   
    
  AzPmatVar mv; data->gen_data(ia_dxs.point(), ia_dxs.size(), mv); 
  AzDmat md; AzpG_Tools::to_img(mv, md, w0*h0*cc, do_pm1); 
  AzBytArr byt; 
  for (int hx = 0; hx < hnum; ++hx) { /* images */
    for (int hpx = 0; hpx < h0; ++hpx) { /* pixels */
      for (int wx = 0; wx < wnum; ++wx) { /* images */
        int col = hx*wnum+wx; 
        for (int wpx = 0; wpx < w0; ++wpx) { /* pixels */
          for (int cx = 0; cx < cc; ++cx) { /* channels */
            AzByte val = (AzByte)md.get((hpx*w0+wpx)*cc + cx, col); 
            byt.concat(val); 
          }
        }
        if (wx < wnum-1) {
          for (int ix = 0; ix < gap*cc; ++ix) byt.concat(255);  
        }
      }
    }
    if (hx < hnum-1) {
      for (int ix = 0; ix < gap; ++ix) {
        for (int jx = 0; jx < wid*cc; ++jx) byt.concat(255); 
      }      
    }
  }
  file.writeBytes(byt.point(), byt.length()); 
  file.close(true); 
}

/*------------------------------------------------------------*/
void AzpG_Tools::order_by_cls(AzpReNet *clsnet, int mb, 
          const AzpData_ *data, 
          AzIntArr &ia_io_dxs, /* inout */
          bool do_entropy, int cls_no, bool do_rev, int each_num) {                              
  const char *eyec = "AzpG_Tools2::order_by_cls"; 
  AzX::no_support(data->batchNum() > 1, eyec, "multiple data batches");   
 
  AzParam clas_azp(""); 
  clsnet->init_test(clas_azp, data);   
  int cls_num = clsnet->classNum(); 
  AzX::throw_if(cls_no >= cls_num, eyec, "The class parameter is out of range.");  
  
  AzIntArr ia_i_dxs(&ia_io_dxs); 
  AzIIarr iia; AzIFarr ifa; 
  AzDataArr<AzIFarr> aIf; 
  for (int pos = 0; pos < ia_i_dxs.size(); pos += mb) {
    int d_num = MIN(mb, ia_i_dxs.size()-pos); 
    AzPmatVar mv_img; 
    data->gen_data(ia_i_dxs.point()+pos, d_num, mv_img); 
    bool is_test = true; 
    AzPmatVar mv_pred; 
    clsnet->up(is_test, mv_img, mv_pred); 
    AzDmat m_pred; mv_pred.data()->get(&m_pred);    
    AzX::throw_if(m_pred.colNum() != d_num, eyec, "Expected one output per data point"); 
    AzDmat m_entropy; 
    if (do_entropy) {
      AzPmat m(mv_pred.data()); m.exp(); m.add(1e-10); m.normalize1();  
      AzPmat m1(&m); m1.log(); m1.elm_multi(&m); m1.multiply(-1); 
      m.sum_per_col(&m1); /* -sum_c p_c ln(p_c) */
      m.get(&m_entropy); 
    }    
    for (int col = 0; col < d_num; ++col) {
      int dx = ia_i_dxs[pos+col]; 
      if      (do_entropy)  ifa.put(dx, m_entropy.get(0, col));
      else if (each_num > 0) {
        if (aIf.size() <= 0) aIf.reset(cls_num); 
        for (int cls = 0; cls < cls_num; ++cls) aIf(cls)->put(dx, m_pred.get(cls, col)); 
      }    
      else if (cls_no >= 0)  ifa.put(dx, m_pred.get(cls_no, col));  
      else {
        int cls = -1;     
        m_pred.col(col)->max(&cls); 
        iia.put(cls, dx);  
      }
    }    
  }    
  ia_io_dxs.reset(); 
  if (iia.size() > 0) { /* sort by class */
    iia.sort(!do_rev); 
    iia.int2(&ia_io_dxs); 
  }
  else if (aIf.size() > 0) {
    for (int cls = 0; cls < aIf.size(); ++cls) {
      aIf(cls)->sort_Float(do_rev); 
      aIf(cls)->cut(each_num); 
      AzIntArr ia; aIf(cls)->int1(&ia); 
      ia_io_dxs.concat(&ia); 
    }
  }
  else if (ifa.size() > 0) { /* sort by score */
    ifa.sort_Float(do_rev);
    ifa.int1(&ia_io_dxs);   
  }
}

/*------------------------------------------------------------*/ 
/*------------------------------------------------------------*/ 
void AzpG_unsorted_images::init(const AzBytArr &s_fn_stem, int gen_num, bool _do_pm1, 
                                 int _wid, int _hei, int _cc) {
  index=0; wid=_wid; hei=_hei; cc=_cc; do_pm1=_do_pm1;   
  if (do_no_collage) {
    s_fn.reset(&s_fn_stem); 
    return; 
  }

  hh = (int)sqrt((double)gen_num); 
  while (gen_num % hh != 0) --hh; 
  ww = gen_num / hh; 
  AzIntArr ia; ia.put(0); ia.put(wid*hei*ww*hh); 
  mv.reform(cc, &ia);     
 
  bytes.reset(); 
  AzBytArr s_ext; 
  AzpG_Tools::gen_ppm_header(ww*wid, hh*hei, cc, bytes, &s_ext); 
  s_fn.reset(s_fn_stem.c_str(), s_ext.c_str());   
}
void AzpG_unsorted_images::proc(const AzPmatVar &mv_gen) {
  if (do_no_collage) {
    AzpG_Tools::to_img_ppm(mv_gen, wid, hei, cc, do_pm1, s_fn.c_str(), index); 
    return; 
  }
  const char *eyec = "AzpG_unsorted_images::proc";  
  AzX::throw_if(mv_gen.rowNum() != cc, eyec, "#channel <> #row?!");  
  AzIntArr ia_scols, ia_dcols; 
  for (int dx = 0; dx < mv_gen.dataNum(); ++dx, ++index) {
    AzX::throw_if(mv_gen.size(dx) != wid*hei, eyec, "unexpected image shape");
    AzX::throw_if(index > ww*hh, eyec, "too many images");
    int w0 = index % ww, h0 = index / ww;
    int col1 = mv_gen.get_begin(dx);
    for (int h1 = 0; h1 < hei; ++h1) {
      for (int w1 = 0; w1 < wid; ++w1, ++col1) {
        ia_scols.put(col1); 
        ia_dcols.put((h0*hei+h1)*(wid*ww) + w0*wid + w1);
      }         
    }
  }
  mv.data_u()->copy_scol2dcol(mv_gen.data(), ia_scols, ia_dcols); 
}
void AzpG_unsorted_images::flush() {
  if (do_no_collage) return; 
  
  AzFile file(s_fn.c_str()); file.open("wb");    
  AzDmat md; AzpG_Tools::to_img(mv, md, mv.size(), do_pm1); 
  for (int row = 0; row < md.rowNum(); ++row) bytes << (AzByte)md.get(row,0); 
  file.writeBytes(bytes.point(), bytes.length());   
  file.close(true); 
}
