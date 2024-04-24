      SUBROUTINE GWKICK(m1,m2,s1,theta1,phi1,s2,theta2,phi2,
     &a1,a2,vperp1,vperp2,vpar,xeff,afin,sfin,thfin)
*
* Gravitational-wave recoil kick and final BH spin 
*
* Lousto, C. O. et al., 2012, Phys. Rev. D., 85, 084015
* Hofmann, F. et al., 2016, ApJ Lett., 825, L19
* 
      implicit none
*
      real*8 m1,m2,s1,theta1,phi1,s2,theta2,phi2,a1,a2,eta,q
      real*8 m, s, theta, phi, ax, fac, termA, termB, cosA
      real*8 termS, termK, atot, aeff, rISCO, LISCO, EISCO, lsum
      real*8 termK1, termK2
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
      real*8 A, B, H, ksi, V11, VA, VB, VC  
******** Lousto, C. O. et al., 2012, Phys. Rev. D., 85, 084015
      parameter(A=1.2D+04,B=-0.93D0,H=6.9D+03,ksi=2.530727415392D0)
      parameter (V11=3677.76D0)
      parameter (VA=2481.21D0, VB=1792.45D0, VC=1506.52D0)
***** A, H, V11, VA, VB, VC in km/sec; B, ksi dimensionless ********
*****
      real*8 ksifac
      integer nm, nj, i, j, imode 
      real*8 kconst(4,5) 
*     parameter (imode=12)
*     parameter (imode=33)
      parameter (imode=34)
****  imode can only be 12, 33, or 34 *****
      do i = 1,4
         do j = 1,5
            kconst(i,j) = 0.0D0
         end do
      end do
      ksifac = 0.0D0
      nm = 0
      nj = 0
      lsum = 0.0D0
*
******* Hofmann, F. et al., 2016, ApJ Lett., 825, L19; Table 1 *******
      if(imode.eq.12) then
        kconst(1,1) = -3.82D0
        kconst(1,2) = -1.2019D0
        kconst(1,3) = -1.20764D0
        kconst(2,1) =  3.79245D0 
        kconst(2,2) =  1.18385D0
        kconst(2,3) =  4.90494D0
        ksifac = 0.41616D0
        nm = 2
        nj = 3
      end if
*
      if(imode.eq.33) then
        kconst(1,1) = -5.9D0
        kconst(1,2) =  2.87025D0
        kconst(1,3) = -1.53315D0
        kconst(1,4) = -3.78893D0
        kconst(2,1) =  32.9127D0
        kconst(2,2) = -62.9901D0
        kconst(2,3) =  10.0068D0
        kconst(2,4) =  56.1926D0
        kconst(3,1) = -136.832D0
        kconst(3,2) =  329.32D0
        kconst(3,3) = -13.2034D0
        kconst(3,4) = -252.27D0
        kconst(4,1) =  210.075D0
        kconst(4,2) = -545.35D0
        kconst(4,3) = -3.97509D0
        kconst(4,4) =  368.405D0
        ksifac = 0.463926D0
        nm = 4
        nj = 4
      end if
*
      if(imode.eq.34) then
        kconst(1,1) = -5.9D0
        kconst(1,2) =  3.39221D0
        kconst(1,3) =  4.48865D0
        kconst(1,4) = -5.77101D0
        kconst(1,5) = -13.0459D0
        kconst(2,1) =  35.1278D0
        kconst(2,2) = -72.9336D0
        kconst(2,3) = -86.0036D0
        kconst(2,4) =  93.7371D0
        kconst(2,5) =  200.975D0
        kconst(3,1) = -146.822D0
        kconst(3,2) =  387.184D0
        kconst(3,3) =  447.009D0
        kconst(3,4) = -467.383D0
        kconst(3,5) = -884.339D0
        kconst(4,1) =  223.911D0
        kconst(4,2) = -648.502D0
        kconst(4,3) = -697.177D0
        kconst(4,4) =  753.738D0
        kconst(4,5) =  1166.89D0
        ksifac = 0.474046D0 
        nm = 4
        nj = 5
      end if
*
      if(imode.ne.12.AND.imode.ne.33.AND.imode.ne.34) then
        write (6,*) "Polynomial not set. Exit."
        STOP
      end if
*******
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
      vperpm = A*(eta**2)*((1.0D0-q)/(1.0D0+q))*(1.0D0+B*eta)
*
* In-plane spin-determined recoil
      vperps = ((H*(eta**2))/(1.0D0+q))
     &*(a2*COS(theta2)-q*a1*COS(theta1))
*
* Off-plane (orthogonal) spin-determined recoil
      termS=(2.0D0*(a2*COS(theta2)+(q**2)*a1*COS(theta1)))/((1+q)**2)
      termA = V11 + (VA*termS) + (VB*(termS**2)) + (VC*(termS**3)) 
      termK1 = a2*SIN(theta2)*COS(phi2) - q*a1*SIN(theta1)*COS(phi1)
      termK2 = a2*SIN(theta2)*SIN(phi2) - q*a1*SIN(theta1)*SIN(phi1)
      termK = SQRT(termK1**2 + termK2**2)
      termB = termK*COS(phi1)
* Lousto, C. O. et al., 2012, Phys. Rev. D., 85, 084015
      vpar = ((16.0D0*(eta**2))/(1.0D0+q))*termA*termB
*
* Orthogonal components of in-plane recoil
      vperp1 = vperpm + vperps*COS(ksi)
      vperp2 = vperps*SIN(ksi)
*
* Effective spin parameter
      xeff = (m1*a1*COS(theta1) + m2*a2*COS(theta2))/(m1 + m2)
*
* Dimensionless spin of the merged BH
* Hofmann, F. et al., 2016, ApJ Lett., 825, L19
* !Note the convention of m1 and m2 is opposite (m1 >= m2)
* in Hofmann et al. Here rewriting them with m1 <= m2!
      cosA = COS(theta2)*COS(theta1)
     & + SIN(theta2)*SIN(theta1)*COS(phi1-phi2)
      termC = a2**2 + (q**4)*(a1**2) + 2.0D0*(q**2)*a2*a1*cosA
      termD = a2*COS(theta2) + (q**2)*a1*COS(theta1)
*     
      atot = termD/((1.0D0 + q)**2)
      aeff = atot + ksifac*eta*(a2*COS(theta2) + a1*COS(theta1))
*
      call ISCO(aeff,rISCO,EISCO,LISCO)
      do i = 1,nm
         do j = 1,nj
            lsum = lsum + kconst(i,j)*(eta**i)*(aeff**(j-1)) 
         end do
      end do
*
      l = LISCO - 2.0D0*atot*(EISCO - 1.0D0) + lsum
      l = DABS(l)
*
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
      END SUBROUTINE GWKICK
*
*
      SUBROUTINE ISCO(a,r,E,L)
*
*     Specific (dimensionless) angular momentum and energy for a
*     test particle at the innermost stable circular orbit (ISCO)
*     of a Kerr BH with spin parameter 'a'
*     Bardeen, J. M., Press, W. H., & Teukolsky, S. A. 1972, ApJ, 178, 347
*     Hofmann, F. et al., 2016, ApJ Lett., 825, L19
*
      implicit none
*
      real*8 a, r, E, L
      real*8 Z1, Z2, termZ, termR
*
      Z1 = 1.0D0 +
     &     ((1.0D0 - a**2)**0.333333D0)*
     &     ((1.0D0 + a)**0.333333D0 + (1.0D0 - a)**0.333333D0)
      Z2 = 3.0D0*(a**2) + Z1**2
      Z2 = SQRT(Z2)
*
      termZ = (3.0D0 - Z1)*(3.0D0 + Z1 + 2.0D0*Z2)
      termZ = SQRT(termZ)
      r = 3.0D0 + Z2 - (a/DABS(a))*termZ
*
      termR = 3.0D0*r - 2.0D0
      termR = SQRT(termR)
      L = (2.0D0/(3.0D0*SQRT(3.0D0)))*(1.0D0 + 2.0D0*termR)
*
      E = 1.0D0 - 2.0D0/(3.0D0*r)
      E = SQRT(E)
*
      return
*
      END SUBROUTINE ISCO
