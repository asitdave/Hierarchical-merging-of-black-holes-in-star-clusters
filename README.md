## Black Hole Tango: Simulating Hierarchical Mergers using Python 

This Python script looks into the hierarchical black hole mergers, meticulously simulating their collisions and analyzing the aftereffects. Buckle up, astrophysicists and coders alike, as we embark on a journey through the cosmos!

**What it Does:**
* Simulates hierarchical merging of black holes in different star cluster environments - Open, Globular, and Nuclear.
* Analyzes key parameters like final black hole mass, kick velocity, Kerr parameter, Effective spin parameter, Spin parameter, and Polar angle.
* Offers functionalities to visualize correlations between these parameters. 
* Can handle escape conditions based on environment (nuclear, young, or globular clusters).

**How it Works:**

1. **Initialization:** 
   - The script read black hole data (mass and Kerr parameter) from `bhlist.dat` file and uses it to select a black hole randomly for a merger. You can make your own set of data, with first column for Black hole mass, and second for Kerr parameter. 
   - It sets up the simulation parameters, including the number of generations and escape criteria.

2. **Black hole Merger:**
   - In each generation, the script simulates the merger of two black holes using an external executable (`gwkik2`).
   - It extracts relevant data from the simulation output, such as the final black hole properties.

3. **Generational Analysis:**
   - The script keeps track of the data across generations, building a comprehensive picture of the merger history.
   - It can calculate the inherent probability for each generation for respective cluster environment.

4. **Visualization (Optional):**
   - The script offers functions to create plots that visualize the evolution of various parameters across generations.
   - These plots can reveal trends and correlations between black hole properties.

**Getting Started:**

1. **Requirements:**
   - Python 3, gfortran
   - External executable `gwkik2`
   
   Follow these steps to make an executable `gwkik2` file:
      1. Download the bh_kick_code folder.
      2. Open a terminal in that folder and type `gfortran-mp-9 -Wall -ffast-math gwkick2.f gwrec.f -o gwkik2` 
      3. Then, make the executable file using the command `chmod +x gwkik2`. This should create the executable file required for this script.

2. **Instructions:**
   - Clone this repository or download the script.
   - Ensure you have the necessary libraries and `gwkik2.exe` file.
   - Make sure that the directory in which you have this python script is the same directory where *bh_kick_code* directory is stored.
   - Edit the script to configure simulation parameters (number of generations, escape conditions, etc.) and file paths. 
   - Before running the script type `python3 BH_merger.py -h (or --help)`

**Beyond the Basics:**

- Modify the script to change the initial parameters of the first two black holes, and the escape velocities of the merger environments.
- Implement additional analysis techniques to delve deeper into the simulation results.
- Feel free to adapt the visualization functions to suit your specific needs.

**Sharing the Love:**

- If you find this script useful, consider giving it a star on GitHub!
- Feel free to contribute to the project by reporting issues or suggesting improvements.

**Further Enhancements:**

- The script can be extended to incorporate more sophisticated black hole merger models.
- It could be integrated with other tools for a more comprehensive analysis pipeline.


**Acknowledgement:**
- This project was guided by Dr. Sambaran Banerjee, University of Bonn, Germany. It was because of his prior work on the black hole mergers that I could use his fortran code, which calculates the black hole parameters post merger.


**References:**
- Aasi, J. et al. 2015. ‘Advanced LIGO’. Classical and Quantum gravity 32 (7): 074001.
- Acernese, F. et al. 2014. ‘Advanced Virgo: a second-generation interferometric gravitational wave detector.’ Classical and Quantum Gravity 32 (2): 024001.
- Boyle, L. et al. 2008. ‘Binary–Black-Hole Merger: Symmetry and the Spin Expan- sion.’ Physical Review Letters 100 (15): 151101.
- Campanelli, M. et al. 2006. ‘Spinning-black-hole binaries: The orbital hang-up’. Physical Review D 74 (4): 041501.
- Campanelli, Manuela et al. 2007. ‘Maximum gravitational recoil’. Physical Review Letters 98 (23): 231102.
- Fishbach, Daniel E. Holz., Maya. 2017. ‘Where are LIGO’s big black holes?’ The Astrophysical Journal Letters 851 (2): L25.
- González, J.A. et al. 2007. ‘Supermassive recoil velocities for binary black-hole mergers with antialigned spins.’ Physical Review Letters 98 (23): 231101.
- Hofmann, Fabian et al. 2016. ‘The final spin from binary black holes in quasi- circular orbits’. The Astrophysical Journal Letters 825 (2): L19.
- Lousto, Carlos O. et al. 2012. ‘Gravitational recoil from accretion-aligned black- hole binaries’. Physical Review D 85 (8): 084015.
- Lousto, Y., C.O. Zlochower. 2011. ‘Hangup kicks: still larger recoils by partial spin-orbit alignment of black-hole binaries.’ Physical Review Letters 107 (23): 231102.
- Schnittman, A., J.D. Buonanno. 2007. ‘The distribution of recoil velocities from merging black holes’. The Astrophysical Journal 662(2) (2): L63.



This script is a springboard for your exploration of hierarchical black hole mergers. Let's unlock the secrets of these cosmic encounters, one simulation at a time!
