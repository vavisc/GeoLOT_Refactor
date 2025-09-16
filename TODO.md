- [ ] Ford-Dataset validieren:
  - [x] GPS-Punkte visualisieren und auf Plausibilität prüfen
  - [x] Kameras validieren (Transformationen, Intrinsics, Bilddimensionen)
  - [x] Bilddaten prüfen (passen Kamerabilder und Luftbilder zusammen?)
  - [x] Alignment zwischen GPS-Punkten und Satellitenbildern verifizieren
  - [x] Abgleich mit Referenzwerten aus "Highly Accurate" Publikation
  - [ ] Write Issue about sorting the txt file

- [ ] Preprocessing-Wrapper entwickeln:
  - [ ] Augmentierungen definieren und konfigurierbar machen
  - [x] Satelliten-Cropping über affine Transformation (Shift + Rotation berücksichtigen)
  - [ ] LIDAR-Einblenden um zu sehen, wie gut gt stimmt.

- [ ] Training
  - [ ] Check if ford_ha sorted does effect performance
  - [ ] Check if KD helps in refining GT -FMAPS could get sharper

- [ ] README
  - [ ] Add section about scripts for visualization of GPS points in train, test splits
