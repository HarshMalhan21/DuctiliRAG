# Metallurgical Failure Analysis & ASTM Standards Reference Guide

## 1. Non-Metallic Inclusions and Subsurface Fish-Eye Fatigue (dA, dB, dC, RedRatio)
In carbon and low-alloy steels subjected to rotating bending fatigue (10^7 cycles), non-metallic inclusions act as severe internal stress raisers:
- **Type A Inclusions (dA - Manganese Sulfides, MnS):** Elongated deformable inclusions formed during hot rolling. They cause mechanical anisotropy and initiate surface or subsurface slip-band fatigue cracks when cyclic stress is transverse to the rolling direction.
- **Type B Inclusions (dB - Alumina Clusters, Al2O3):** Hard, brittle, discontinuous oxide stringers. Under high-cycle rotating bending fatigue, alumina inclusions nucleate subsurface 'fish-eye' fatigue fractures due to elastic stiffness mismatch and localized tensile stress concentration.
- **Type C Inclusions (dC - Silicates and Globular Oxides):** Undeformable globular oxide inclusions that cause matrix-inclusion decohesion and micro-void nucleation under cyclic loading.
- **Forging Reduction Ratio (RedRatio):** A low hot-working reduction ratio (< 500) leaves unclosed casting microporosity, coarse dendritic segregation, and large inclusion clusters, drastically reducing rotating bending endurance limits.
- **Applicable Standard - ASTM E45 & JIS G 0555:** Standard Test Methods for Determining the Inclusion Content of Steel using microscopic rating charts (Methods A and D).
- **Recommended Mitigation:** Secondary steelmaking refinement via vacuum degassing, calcium wire injection for sulfide shape control, or Electro-Slag Remelting (ESR) to eliminate coarse Type B alumina clusters.

## 2. Hardenability, Carbon Equivalent (CE), and Alloying Elements (C, Cr, Mo, Ni, Mn, Si)
Rotating bending fatigue strength in steels correlates linearly with matrix hardness and ultimate tensile strength (UTS) up to approximately 400 HB (~1400 MPa UTS):
- **Carbon Equivalent (CE) and Carbon (C):** Low Carbon (< 0.35% C) or low Carbon Equivalent (CE < 0.55) limits maximum achievable martensite hardness and promotes formation of soft proeutectoid ferrite during quenching. Soft ferrite grains undergo early cyclic plastic slip (persistent slip bands) at stress amplitudes below 450 MPa.
- **Chromium (Cr) and Molybdenum (Mo):** Insufficient Cr (< 0.4%) reduces core hardenability and resistance to carbide coarsening during tempering. Absence of Molybdenum (Mo = 0, Mo_added = 0) leaves prior austenite grain boundaries susceptible to phosphorus segregation and **Temper Embrittlement** when tempered or cooled slowly through 350-575°C, promoting low-energy intergranular fracture.
- **Nickel (Ni) and Silicon (Si):** Nickel enhances cross-slip in ferrite, increasing matrix fracture toughness and lowering the ductile-to-brittle transition temperature. Silicon strengthens ferrite by solid-solution hardening and delays cementite coarsening during tempering.
- **Applicable Standard - ASTM A255:** Standard Test Methods for Determining Hardenability of Steel (Jominy End-Quench Test).

## 3. Heat Treatment Defects in Normalizing, Quenching, and Tempering (NT, THT, THQCr, TT, Tt)
Improper thermal processing degrades the fatigue limit of medium-carbon and Cr-Mo alloy steels:
- **Normalizing Temperature (NT):** Improper normalizing temperature fails to eliminate banded ferrite-pearlite microstructures or causes heterogeneous prior-austenite grain growth prior to final hardening.
- **Quench Cooling Rate (THQCr) and Quench Media Temperature (QmT):** A low cooling rate (THQCr <= 8 °C/s) fails to bypass the pearlite/bainite nose of the CCT diagram, producing slack-quenched upper bainite or grain-boundary ferrite networks at the surface instead of 100% lath martensite. Slack-quenched microstructures reduce rotating bending fatigue strength by 25-40%.
- **Tempering Temperature (TT) and Time (Tt):** Over-tempering (TT >= 550-600°C or prolonged Tt >= 60 min) in plain carbon or low-Cr/Mo steels causes rapid recovery of the martensitic dislocation substructure and Ostwald ripening (spheroidization) of cementite (Fe3C) particles, dropping endurance limits below 450 MPa. Conversely, tempering at 250-400°C triggers Tempered Martensite Embrittlement (TME).
- **Applicable Standard - ASTM E18 & ASTM E384:** Standard Test Methods for Rockwell and Vickers Microindentation Hardness of Materials.

## 4. Carburizing and Case-Hardening Failure Mechanisms (Carburized, CT, Ct, Dt)
Rotating bending imposes maximum alternating tensile-compressive stress at the outer surface:
- **Uncarburized Condition (Carburized = 0):** Non-carburized steels lack both the high-hardness martensitic case (~60 HRC) and the deep compressive residual stress field (-200 to -450 MPa) created by case hardening, leaving the surface vulnerable to slip-band, scratch, and fretting crack initiation.
- **Improper Case Depth (CT, Ct, Dt):** Insufficient carburizing time (Ct) or diffusion time (Dt) creates a shallow effective case depth, leading to subsurface 'case crushing' (spalling) at the case-core transition zone where shear stress exceeds core yield strength. Excessive carburizing temperature (CT > 920°C) causes excessive retained austenite and internal grain-boundary oxidation (IGO).
- **Applicable Standard - ASTM E1077 & SAE J423:** Standard Test Methods for Estimating Depth of Decarburization and Effective Case Depth in Steels.

## 5. Fatigue Testing and Non-Destructive Evaluation (NDE) Standards
- **ASTM E466:** Standard Practice for Conducting Force Controlled Constant Amplitude Axial and Rotating-Bending Fatigue Tests of Metallic Materials (High-Cycle Fatigue, 10^6 to 10^7 cycles).
- **ASTM E606:** Standard Test Method for Strain-Controlled Fatigue Testing.
- **ASTM E647:** Standard Test Method for Measurement of Fatigue Crack Growth Rates (da/dN).
- **ASTM E399:** Standard Test Method for Linear-Elastic Plane-Strain Fracture Toughness (K_Ic).
- **ASTM E1444 / ASTM E709:** Standard Practice for Magnetic Particle Testing (MPI) for detecting surface and near-surface fatigue cracks in ferromagnetic steels.
