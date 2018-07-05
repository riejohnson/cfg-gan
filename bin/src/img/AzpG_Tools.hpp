/* * * * *
 *  AzpG_Tools.hpp
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
#ifndef _AZPG_TOOLS_HPP_
#define _AZPG_TOOLS_HPP_

#include "AzDmat.hpp"
#include "AzPmat.hpp"
#include "AzParam.hpp"
#include "AzPrint.hpp"
#include "AzpData_.hpp"
#include "AzpReNet.hpp"

class AzpG_Tools {
public:
  static void to_img_bin(const AzPmatVar &mv, int width, int height, int cc, 
                        bool do_pm1, AzFile *ofile); 
  static void to_img_ppm(const AzPmatVar &mv, int width, int height, int cc, 
                         bool do_pm1, const char *fn_stem, int dx_begin, 
                         const AzStrPool *sp=NULL);                   
  static void to_img(const AzPmatVar &mv, AzDmat &md, int sz, bool do_pm1);
  static void gen_collage_ppm(const AzpData_ *data, bool do_pm1, 
          const AzIntArr &ia_dxs, int gap, const char *out_fn, int wnum, int hnum); 
  static void order_by_cls(AzpReNet *clsnet, int mb,
          const AzpData_ *data, AzIntArr &ia_io_dxs, 
          bool do_entropy=false, int cls_no=-1, bool do_rev=false, int each_num=-1); 
  static void gen_ppm_header(int width, int height, int cc, 
                             AzBytArr &s_head, AzBytArr *s_ext=NULL); 
}; 

/*------------------------------------------------------------*/ 
class AzpG_unsorted_images {
protected:
  bool do_no_collage; 
  int ww, hh, index, wid, hei, cc; 
  AzPmatVar mv; 
  bool do_pm1; 
  AzBytArr bytes, s_fn; 
public:   
  AzpG_unsorted_images(const AzBytArr &s_fn_stem, int gen_num, bool _do_pm1, 
                        int _wid, int _hei, int _cc, bool _do_no_collage=false)
    : ww(-1), hh(-1), index(0), wid(-1), hei(-1), cc(-1), do_pm1(false), do_no_collage(false) {
    do_no_collage = _do_no_collage; 
    init(s_fn_stem, gen_num, _do_pm1, _wid, _hei, _cc); 
  }
  void init(const AzBytArr &s_fn_stem, int gen_num, bool _do_pm1, int _wid, int _hei, int _cc); 
  void proc(const AzPmatVar &mv_gen); 
  void flush(); 
}; 
#endif 
