      PROGRAM RANGEN
*
*
*     A simple wrapper of NR's ran2 RNG
*     ---------------------------------
*
      INTEGER IDUM1, KDUM 
      REAL*8 RAN2, RN1
*
      read(5,*) IDUM1
*
*       Initialize the portable random number generator (range: 0 to 1).
      KDUM = -1
      RN1 = RAN2(KDUM)
*       Skip the first random numbers (IDUM1 specified at input).
      DO 1 K = 1,IDUM1
          RN1 = RAN2(KDUM)
    1 CONTINUE

*      Output next random number
      RN1 = RAN2(KDUM)
      WRITE(6,10) RN1
   10 FORMAT (F12.10)
*
      RETURN
*
      END
