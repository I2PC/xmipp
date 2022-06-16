/***************************************************************************
 *
 * Authors:     Edgar Garduno Angeles (edgargar@ieee.org)
 * Created on: Sep 6, 2019
 *
 * Department of Computer Science, Institute for Applied Mathematics
 * and Systems Research (IIMAS), National Autonomous University of
 * Mexico (UNAM)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/
#ifndef SUPERIORIZATION_REGULARIZER_HH
#define SUPERIORIZATION_REGULARIZER_HH

#include <functional>
#include <core/xmipp_program.h>

#include "second_criterion.h"
#include "aniso_total_variation.h"
#include "iso_total_variation.h"
#include "w_total_variation.h"

// template <typename R, typename ...ARGS> using function = R(*)(ARGS...);
// template< class R, class... Args > class function<R(Args...)>;
template<class T>
class SuperRegular: public MultidimArray<T>
{
 public:
	enum classType{none,ITV,WTV,ATV};

 private:
 //
 // Methods
 //

 public:
	SuperRegular();
	SuperRegular(String StrType);
	void set(std::string StrType);
	double operator ()(const MultidimArray<T>& x);
	MultidimArray<T> nav(const MultidimArray<T>& x);
	void init(MultidimArray<T>& x);
	void init(MultidimArray<double>& u, const double sigma, const double ku,
                  const double me, const double delta,
                  const double Amin, const double Amax);
	void preupdate(MultidimArray<T>& x);
	void postupdate(MultidimArray<T>& x);
	bool valid(const String &StrType);
	String getName(void);
	String getShortName(void);
 protected:

 private:
    classType RegType;
    secc *SecC;
};

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/****************** Definition of Class Methods ********************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

/**
**
** Basic constructor.
**
*/
template<class T>
SuperRegular<T>::SuperRegular()
{
 //phi = std::bind(&itv::tv,this,std::placeholders::_1);
 //nav = std::bind(&itv::vtv,this,std::placeholders::_1,std::placeholders::_2);
 SecC = 0;
 SecC = new itv;
 RegType = SuperRegular::ITV;
}

/**
**
** Constructor from a name (String).
**
*/
template<class T>
SuperRegular<T>::SuperRegular(String StrType)
{
 if(StrType == std::string("ATV")){
    SecC = new atv;
    RegType = SuperRegular::ATV;
   }
 if(StrType == std::string("ITV")){
    SecC = new itv;
    RegType = SuperRegular::ITV;
   }
 if(StrType == std::string("WTV")){
    SecC = new wtv;
    RegType = SuperRegular::WTV;
   }
}

/**
**
** Method to set the desired function to be used as a second criterion.
**
*/
template<class T>
void SuperRegular<T>::set(std::string StrType)
{
 if(SecC != 0)
    delete SecC;
 
 if(StrType == std::string("ATV")){
    SecC = new atv;
    RegType = SuperRegular::ATV;
   }
 if(StrType == std::string("ITV")){
    SecC = new itv;
    RegType = SuperRegular::ITV;
   }
 if(StrType == std::string("WTV")){
    SecC = new wtv;
    RegType = SuperRegular::WTV;
   }
}

/**
**
** Method to set the desired function to be used as a second criterion.
**
*/
template<class T>
double SuperRegular<T>::operator ()(const MultidimArray<T>& x)
{
 return SecC->phi(x);
}

/**
**
** Method to set the desired function to be used as a second criterion.
**
*/
template<class T>
MultidimArray<T> SuperRegular<T>::nav(const MultidimArray<T>& x)
{
 return SecC->nav(x);
}

/**
**
** Method to set the desired function to be used as a second criterion.
**
*/
template<class T>
void SuperRegular<T>::init(MultidimArray<T>& x)
{
 SecC->init(x);
}

/**
**
** Method to set the desired function to be used as a second criterion.
**
*/
template<class T>
void SuperRegular<T>::init(MultidimArray<double>& u, const double sigma, const double ku, const double me, const double delta, const double Amin, const double Amax)
{
 SecC->init(u,sigma,ku,me,delta,Amin,Amax);
}

/**
**
** Method to execute the method to pre-update the data for the second criterion.
**
*/
template<class T>
void SuperRegular<T>::preupdate(MultidimArray<T>& x)
{
 SecC->preupdate(x);
}

/**
**
** Method to execute the method to post-update the data for the second criterion.
**
*/
template<class T>
void SuperRegular<T>::postupdate(MultidimArray<T>& x)
{
 SecC->postupdate(x);
}

/**
**
** Method to check whether the inquired function is valid.
**
*/
template<class T>
bool SuperRegular<T>::valid(const String &StrType)
{
 if(StrType=="ATV")
    return true;

 if(StrType=="ITV")
    return true;

 if(StrType=="WTV")
    return true;

 return false;
}

/**
**
** Method to set the desired function to be used as a second criterion.
**
*/
template<class T>
String SuperRegular<T>::getName(void)
{
 switch(RegType){
     case ATV:
         return "anisotropic TV";
         break;
     case ITV:
         return "isotropic TV";
         break;
     case WTV:
         return "weighted TV";
         break;
     default:
         return "No Second Criterion";
    }

 return "No Second Criterion";
}

/**
**
** Method to set the desired function to be used as a second criterion.
**
*/
template<class T>
String SuperRegular<T>::getShortName(void)
{
 switch(RegType){
     case ATV:
         return "ATV";
         break;
     case ITV:
         return "ITV";
         break;
     case WTV:
         return "WTV";
         break;
     default:
         return "No Second Criterion";
    }

 return "No Second Criterion";
}

#endif /* SUPERIORIZATION_REGULARIZER_HH */
