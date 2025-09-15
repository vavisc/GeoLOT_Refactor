- [ ] Ford-Dataset validieren:
  - [x] GPS-Punkte visualisieren und auf Plausibilität prüfen
  - [x] Kameras validieren (Transformationen, Intrinsics, Bilddimensionen)
  - [x] Bilddaten prüfen (passen Kamerabilder und Luftbilder zusammen?)
  - [x] Alignment zwischen GPS-Punkten und Satellitenbildern verifizieren
  - [x] Abgleich mit Referenzwerten aus "Highly Accurate" Publikation
  - [ ] Write Issue about sorting the txt file

- [ ] Preprocessing-Wrapper entwickeln:
  - [ ] Augmentierungen definieren und konfigurierbar machen
  - [ ] Satelliten-Cropping über affine Transformation (Shift + Rotation berücksichtigen)
  - [ ] Rotationen konsistent anwenden (inkl. Bodyframe des Fahrzeugs)

- [ ] Training
  - [ ] Check if ford_ha sorted does effect performance
