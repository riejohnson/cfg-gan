/* * * * *
 *  AzpReNetG.hpp
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
 
#ifndef _AZP_RE_NET_G_HPP_
#define _AZP_RE_NET_G_HPP_

#include "AzpReNet.hpp"
#include "AzpReLayerG.hpp"

/*------------------------------------------------------------*/
class AzpReNetG : public virtual AzpReNet {
protected:
  AzpReLayersG laysG; 
  
  virtual AzpReNet *clone_nocopy() const { 
    return new AzpReNetG(cs); 
  }   
  
public:
  AzpReNetG(const AzpCompoSet_ *_cs) : AzpReNet(_cs) {
    lays = &laysG; 
  }    
  virtual ~AzpReNetG() {} 
}; 
#endif
