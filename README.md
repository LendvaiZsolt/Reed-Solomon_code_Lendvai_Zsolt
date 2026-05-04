# Projekt dokumentáció: Reed-Solomon hibajavító kódolás bemutató alkalmazás

Ez a projekt három Streamlit oldalon mutatja be a Reed-Solomon kódolást, hiba-injektálást, szindrómaszámítást valamint a dekódolást és javítást:

- `RS(7,4)` oldal (`app.py` -> `rs74_app.py`)
- `RS(8,4)` oldal (`pages/2_rs84.py`)
- `Véletlen hibateszt` oldal (`pages/4_veletlen_hibateszt.py`)

## Indítás

1. Nyisd meg a projekt gyökérmappáját.
2. Terminálban futtasd: `py -m streamlit run "app.py"`
3. A böngészőben megnyílik az alkalmazás.
4. Közvetlen webes elérés (telepítés nélkül): https://reed-solomoncodelendvaizsolt-qcuzfobz6xfm4gpmof7kfb.streamlit.app/

---

## Használati útmutató

### 1. Cél, hogy az alkalmazás bemutassa:

- a kódolást (`m -> c`),
- a csatornahibát (`r`, `e = r - c`),
- a szindrómákat,
- és a javítás/dekódolás lépéseit.

### 2. RS típus kiválasztása

2.1 Alapértelmezett oldal: RS(7,4).  
2.2 Oldalváltással elérhető: RS(8,4) GF(9).  
2.3 Oldalváltással elérhető: Véletlen hibateszt.

### 3. Paraméterek megadása

3.1 RS(7,4)  
3.1.1 Bemenet: 3 betű szimbólum (A-H), a 4. hely automatikus padding (A).  
3.1.2 Generátor mátrix elrendezés: Balra ([p0,p1,p2 | m0,m1,m2,m3]) vagy Jobbra ([m0,m1,m2,m3 | p0,p1,p2]).  
3.1.3 Hiba injektálás: Igen / Nem.

HA TÖRTÉNIK HIBA INJEKTÁLÁS

3.1.4 Hiba módja: Közvetlen fogadott érték (r[j] megadása), vagy Összeadásos mód (r[j] = c[j] + e).  
3.1.5 Hibák száma: 1  
3.1.6 Hibahely (j) és szükség esetén hibaérték (e) megadása minden hibához.

3.2 RS(8,4)  
3.2.1 Bemenet: 4 betű szimbólum (A-P).  
3.2.2 Hiba injektálás: Igen / Nem

HA TÖRTÉNIK HIBA INJEKTÁLÁS

3.2.3 Hiba módja: Közvetlen fogadott érték (r[j] megadása), vagy Összeadásos mód (r[j] = c[j] + e).  
3.2.4 Hibák száma: 1 vagy 2.  
3.2.5 Hibahely (j) és szükség esetén hibaérték (e) megadása minden hibához.  
3.2.6 Előre definiált, illetve 1 vagy 2 random hibás esetek.

### 4. Eredmények értelmezése a füleken

4.1 `Alapadatok`: G és H mátrixok, kódparaméterek.  
4.2 `Kódolás`: `c = m·G` számolás és reprezentációk.  
4.3 `Fogadott szó és hiba`: küldött/fogadott szó, hibavektor.  
4.4 `Szindróma`: hibadetektálás, szindrómák levezetése.  
4.5 `Javítás / dekódolás`: hibahelyreállítás és ellenőrzés.

### 5. Felső lenyitható részek

5.1 Mindkét oldalon a felső, lenyitható részekben folyamatosan ellenőrizhetők a testelemek és az alapműveletek.  
5.2 Itt látható az elem-megfeleltetés (karakter <-> bit <-> polinom <-> alfa-hatvány), valamint az összeadás/szorzás táblázat, ezért kézi ellenőrzéshez ez az elsődleges referencia.

### 6. Korlátok

6.1 `RS(7,4)` legfeljebb **1** szimbólumhibát javít.  
6.2 `RS(8,4)` legfeljebb **2** szimbólumhibát javít.

### 7. Véletlen hibateszt futtatása

A Véletlen hibateszt menü a bal oldali menü utolsó előtti eleme. Egy gombbal lefuttat több véletlen próbát: soronként kiválaszt egy ágat (súlyozva 2-szer akkora eséllyel az RS(8,4) GF(9) ágat), majd ugyanolyan véletlen beállításokkal kódol, injektál hibát és javít, mint a fő oldalak bal oldalánál használt „random” eset kiválasztók. Az eredmény tabulátor szeparált fájlba és táblázatos előnézetbe kerül. A TEST_TRUE azt jelenti, hogy a javított kódszó megegyezik a küldött c kódszóval.

### 8. Ajánlott demonstrációs sorrend

8.1 Hibamentes eset (r=c) referenciának mindkét ágon.  
8.2 Egyhibás esetek mindkét ágon (szindróma → javítás). Beállított dolgozati érték RS(7,4) esetében. Mintahibák és random hibák RS(8,4) esetében.  
8.3 Két hibás esetek az RS(8,4) ágon. Mintahibák és random hibák RS(8,4) esetében.  
8.4 Véletlen hibateszt futtatása 10–50 esetre.

### 9. RS(8,4) — három szimbólumhiba, egyedi legközelebbi kódszó

Bruteforce mintavételezéssel készült tábla: olyan **c** és **r** párok, ahol három szimbólumhiba mellett is az eredeti **c** marad az egyetlen legközelebbi kódszó (elméleti ML-javíthatóság). A 4. oszlopban csak az első négy, távolság 4-es „szomszéd” kódszó látszik, a többi helyén **…** áll.

Nem végeztünk teljes körű vizsgálatot, csak szúrópróbaszerűen ellenőriztük a kódokat. Egy korábbi mérésünk alapján (ahol kb. 90 ezer véletlen próbálkozásból 1568 volt sikeres) a javítási arány nagyjából 1–2% körül mozog. Ez azt jelenti, hogy ha egy tetszőleges kódszóhoz véletlenszerűen három hibát adunk, az esetek csupán töredékében kapjuk vissza egyértelműen az eredeti szót.

[RS(8,4) ML tábla (Markdown a GitHubon)](https://github.com/LendvaiZsolt/Reed-Solomon_code_Lendvai_Zsolt/blob/main/exports/rs84_ml3_unique_nearest_pairs.md)

Verziószám: v1.21 (2026-05-04 03:54:31 budapesti helyi idő, Europe/Budapest; 05c4709)

**GitHub:** [LendvaiZsolt/Reed-Solomon_code_Lendvai_Zsolt](https://github.com/LendvaiZsolt/Reed-Solomon_code_Lendvai_Zsolt)

