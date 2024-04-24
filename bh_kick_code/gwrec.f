      PROGRAM GWREC
*****************
* Determine recoil velocity due to
* anisotropy in GW emission during a binary
* black hole merger.
* Determine final spin of the merged BH. 
* Based on fitting formulae obtained from
* numerical relativity.
* See 'gwkick.f' for references.
*
* Sambaran Banerjee, Bonn, March 2019
*****************
*
      implicit none
*
      real*8 m1, m2, s1, theta1, phi1, s2, theta2, phi2
      real*8 a1, a2, vperp1, vperp2, vpar, xeff, afin, sfin, thfin
      real*8 fac
**************************
* mi (input): BH mass [M_sun] (m2 >= m1)
*
* si,thetai,phii (input): BH spin vector; si [Msun Rsun^2 / year],
* thetai, phii [deg]
* ai (input/output): dimensionless magnitude of BH spin (0<=ai<=1)
* [si >=0.0, ai is evaluated and overwritten,
* si < 0.0, input ai is used]
*
* vperpi (output): in-plane orthogonal components of GW recoil
* [vperp1 along the line joining the merging BHs]
* vpar (output): off-plane GW recoil
*
* xeff (output): effective spin parameter
*
* afin (output): final dimensionless spin magnitude of the merged BH    
* sfin (output): final spin magnitude of merged BH [Msun Rsun^2/year]
* thfin (output): inclination of the merged BHs' spin [deg]
**************************
      real*8 twopi 
      parameter (twopi=6.28318530718D0)
*
      read(5,*) m1, m2, s1, theta1, phi1, s2, theta2, phi2, a1, a2
*
      fac= twopi/360.0D0
      theta1 = fac*theta1
      phi1 = fac*phi1
      theta2 = fac*theta2
      phi2 = fac*phi2
*
      CALL GWKICK(m1,m2,s1,theta1,phi1,s2,theta2,phi2,
     &a1,a2,vperp1,vperp2,vpar,xeff,afin,sfin,thfin)
*
      write (6,*) "BH1,2: m s theta phi a"
      write (6,11) m1, s1, theta1/fac, phi1/fac, a1
      write (6,11) m2, s2, theta2/fac, phi2/fac, a2
      write (6,*)
      write (6,*) "Merged BH: vprp1 vprp2 vpar xeff afin sfin thfin"
      write (6,12) vperp1, vperp2, vpar, xeff, afin, sfin, thfin/fac
   11 format(1P,2E12.2,0P,2F12.3,F12.6)
   12 format(3F16.6,2F12.6,1P,E12.2,0P,F12.3)
*
      return
*
      END
