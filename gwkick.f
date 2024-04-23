      SUBROUTINE GWKICK(m1,m2,s1,theta1,phi1,s2,theta2,phi2,
     &a1,a2,vperp1,vperp2,vpar,xeff,afin,sfin,thfin)
*
* Gravitational-wave recoil kick and final BH spin 
*
* Baker, J. G. et al., 2008, ApJ, 682, L29
* van Meter, J. R. et al., 2010, ApJ, 719, 1427
* Rezzolla, L. et al., 2008, Phys. Rev. D, 78, 044002
* 
      implicit none
*
      real*8 m1,m2,s1,theta1,phi1,s2,theta2,phi2,a1,a2,eta,q
      real*8 m, s, theta, phi, ax, fac, termA, termB, cosA
      real*8 vperpm, vperps, vpar, vperp1, vperp2, xeff, afin
      real*8 sfin, thfin, termC, termD, termF, l, afinp
*
      real*8 G, M_sun, R_sun, parsec, Km, Kmps, cspeed, year
*********** c.g.s. **********************************
      parameter (G=6.6743D-08, M_sun=1.9884D+33)
      parameter (R_sun=6.955D+10, parsec=3.0856776D+18)
      parameter (Km=1.0D+05, Kmps=1.0D+05)
      parameter (cspeed=3.0D+10, year=3.154D+07)
*********
      real*8 A, B, H, ksi, K, K2, K3, Ks  
******** Baker, J. G. et al., 2008, ApJ, 682, L29
      parameter(A=1.35D+04,B=-1.48D0,H=7540.0D0,ksi=3.752457891788D0)
      parameter (K=2.4D+05)
******** van Meter, J. R. et al., 2010, ApJ, 719, 1427 (off-plane kick)
      parameter (K2=32092.0D0, K3=108897.0D0, Ks=15375.0D0)
***** A, B, H, K, K2, K3, Ks in km/sec ********
      real*8 t0, t2, t3, s4, s5
***** Rezzolla, L. et al., 2008, Phys. Rev. D, 78, 044002 (final spin)
      parameter (t0=-2.686D0, t2=-3.454D0, t3=2.353D0)
      parameter (s4=-0.129D0, s5=-0.384D0)
********
*
* Convention: m1 <= m2
      if (m2.lt.m1) then
          m = m2
          s = s2
          theta = theta2
          phi = phi2
          ax = a2
          m2 = m1
          s2 = s1
          theta2 = theta1
          phi2 = phi1
          a2 = a1
          m1 = m
          s1 = s
          theta1 = theta
          phi1 = phi
          a1 = ax
      endif
*
      q = m1/m2
      eta = (m1*m2)/((m1+m2)**2)
*
* Determine dimensionless spin
* (input: m in M_sun, s in Msun Rsun^2 / year)
* c/G [gm cm^-2 sec] -> [M_sun R_sun^-2 year]
      fac=(cspeed/G)*(R_sun**2/(M_sun*year))
* If s1,s2 >=0 evaluate a1,a2
* else use input a1,a2 (must be 0 <= a1,a2 <= 1)
      if ((s1.ge.0.0D0).and.(s2.ge.0.0D0)) then
         a1 = fac*(s1/m1**2)
         a2 = fac*(s2/m2**2)
      end if
*
* Prevent BH spins to become unphysical due to accretion spin-up
* or high spin of the parent star
      if (a1.gt.1.0D0) a1=1.0D0
      if (a2.gt.1.0D0) a2=1.0D0
*
* In-orbital-plane mass-ratio-determined recoil
* (input: theta, phi in radian)
      vperpm = A*(eta**2)*SQRT(1.0D0-4.0D0*eta)*(1.0D0+B*eta)
*
* In-plane spin-determined recoil
      vperps = ((H*(eta**2))/(1.0D0+q))
     &*(a2*COS(theta2)-q*a1*COS(theta1))
*
* Off-plane (orthogonal) spin-determined recoil
      termA = q*a1*SIN(theta1)*COS(phi1) - a2*SIN(theta2)*COS(phi2)
      termB=(q**2)*a1*SIN(theta1)*COS(phi1)+a2*SIN(theta2)*COS(phi2)
* van Meter, J. R. et al., 2010, ApJ, 719, 1427
      vpar = ((K2*(eta**2)+K3*(eta**3))*termA)/(q+1.0D0)
     & + (Ks*(q-1.0D0)*(eta**2)*termB)/((q+1.0D0)**3)
* Baker, J. G. et al., 2008, ApJ, 682, L29
*     vpar = (K*(eta**3)*termA)/(1.0D0+q)
*
* Orthogonal components of in-plane recoil
      vperp1 = vperpm + vperps*COS(ksi)
      vperp2 = vperps*SIN(ksi)
*
* Effective spin parameter
      xeff = (m1*a1*COS(theta1) + m2*a2*COS(theta2))/(m1 + m2)
*
* Dimensionless spin of the merged BH
* Rezzolla, L. et al., 2008, Phys. Rev. D, 78, 044002
* !Note the convention of m1 and m2 is opposite (m1 >= m2)
* in Rezzolla et al!
      cosA = COS(theta2)*COS(theta1)
     & + SIN(theta2)*SIN(theta1)*COS(phi1-phi2)
      termC = a2**2 + (q**4)*(a1**2) + 2.0D0*(q**2)*a2*a1*cosA
      termD = a2*COS(theta2) + (q**2)*a1*COS(theta1)
      l = (s4*termC)/((1.0D0 + q**2)**2)
     & + ((s5*eta+t0+2.0D0)*termD)/(1.0D0+q**2) + 2.0D0*SQRT(3.0D0)
     & + t2*eta + t3*(eta**2)
      termF = termC + (q**2)*(l**2) + 2.0D0*q*l*termD
      afin = SQRT(termF)/((1.0D0 + q)**2) 
*
* Spin angular momentum of merged BH [Msun Rsun^2 / year]
* (GW radiation mass loss neglected)
      sfin = (1.0D0/fac)*afin*((m1+m2)**2) 
*
* Inclination of the merged BHs' spin w.r.t. orbital ang. mom.  
* Rezzolla, L. et al., 2008, Phys. Rev. D, 78, 044002
      afinp = (termD + q*l)/((1.0D0 + q)**2)
      thfin = afinp/afin
      thfin = ACOS(thfin)
*
      return
*
      END
