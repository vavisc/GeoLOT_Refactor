- [ ] Ford-Dataset validieren:
  - [x] GPS-Punkte visualisieren und auf Plausibilität prüfen
  - [x] Kameras validieren (Transformationen, Intrinsics, Bilddimensionen)
  - [x] Bilddaten prüfen (passen Kamerabilder und Luftbilder zusammen?)
  - [x] Alignment zwischen GPS-Punkten und Satellitenbildern verifizieren
  - [x] Abgleich mit Referenzwerten aus "Highly Accurate" Publikation
  - [ ] Write Issue about sorting the txt file

- [ ] Preprocessing-Wrapper entwickeln:
  - [ ] Rotation, Flipping, ColorJitter, Clahe
  - [x] Resize und Normalization
  - [x] Satelliten-Cropping über affine Transformation (Shift + Rotation berücksichtigen)
  - [x] Grid Berechnen und Prüfen
  - [ ] Grid direkt im preprocess wrapper fertig definieren - Derzeit muss es im modell gemacht werden
  - [ ] Grid auch für unterschiedliche Höhen berechnen (see cvgl) 
  - [ ] LIDAR-Einblenden um zu sehen, wie gut gt stimmt.

- [ ] Training
  - [ ] Check if ford_ha sorted does effect performance
  - [ ] Check if KD helps in refining GT -FMAPS could get sharper
  - [ ] Use some sort of flow field transformation in PV to BEV to correct for bad calibration
  - [ ] Perform Grid computation in PV-To-BEV - Allows to later learn flowfields and corrections for intriniscs extrinsics
- [ ] README
  - [ ] Add section about scripts for visualization of GPS points in train, test splits


- [ ] Nächste Schritte:
  - [ ] Modell CVGL aufbauen
    - [ ] ENCODER
    - [ ] PV-To-BEV
    - [ ] Matcher
    - [ ] Loss
  - [ ] PL-Wrapper wo CVGl, aber auch mein Modell passt
  - [ ] Configs mit pydantic