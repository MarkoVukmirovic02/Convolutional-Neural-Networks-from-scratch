# Osnovna ideja neuronskih mreža


Pretpostavimo da imamo ulaznu sliku dimenzija \(n \times 12288\), gde je  
\(12288 = 64 \times 64 \times 3\) (RGB pikseli). Svaka vrednost piksela je u opsegu \([0, 255]\), a pre ulaska u model se **normalizuje** (npr. podelom sa 255).  

Radi jednostavnosti, posmatraćemo **jednu sliku**.  
Dakle, ulazni vektor ima dimenzije:

$
X \in \mathbb{R}^{12288 \times 1}
$

Na osnovu ovog vektora želimo da model predvidi šta se nalazi na slici.  
To može biti **binarna klasifikacija** (npr. mačka / ne-mačka) ili **multiklasna** (npr. više različitih objekata).  

---

## Neuronski slojevi

Prvo definišemo **slojeve**. Broj slojeva i broj neurona u svakom sloju biramo sami.  
Tipične veličine skrivenih slojeva su 64, 128, 256, 512 neurona, itd.  

Pretpostavimo da naš **prvi skriveni sloj** ima 256 neurona.  

Svaki neuron u sloju prima sve ulazne podatke, ali sa različitim **težinama**.  
Težine označavamo sa \(W\), a dodajemo i **bias** \(B\).  
Linearna kombinacija za prvi sloj glasi:

$$
Z^{[1]} = W^{[1]} X + B^{[1]}
$$

**Dimenzije su:**

- $W^{[1]} \in \mathbb{R}^{256 \times 12288}$  
- $X \in \mathbb{R}^{12288 \times 1}$  
- $B^{[1]} \in \mathbb{R}^{256 \times 1}$  
- Rezultat: $Z^{[1]} \in \mathbb{R}^{256 \times 1}$


## Aktivacione funkcije

Na dobijeni vektor $Z^{[1]}$ primenjujemo **aktivacionu funkciju** $\Phi$ da bismo uneli nelinearnost:

$$
A^{[1]} = \Phi(Z^{[1]})
$$
Gde je $\Phi$ najčešće:

- **ReLU**: $\max(0, z)$  
- **tanh**: $\frac{e^z - e^{-z}}{e^z + e^{-z}}$  
- **sigmoid**: $\frac{1}{1 + e^{-z}}$

Rezultat $A^{[1]}$ se prosleđuje sledećem sloju:


$Z^{[2]} = W^{[2]} A^{[1]} + B^{[2]}$


i tako dalje sve do izlaznog sloja.

---

## Izlazni sloj

- Za **binarne klasifikacije** obično koristimo **sigmoid** funkciju.  
- Za **multiklasne klasifikacije** koristi se **softmax** funkcija.  

Ona pretvara linearni izlaz u verovatnoće koje se sabiraju na 1.  

---

### 🔹 Binarna klasifikacija

Funkcija greške za jedan primer:

$
\text{error}^{(i)} = -\Big[y^{(i)}\log(a^{(i)}) + (1-y^{(i)})\log(1-a^{(i)})\Big]
$

Funkcija troška (cost) za celu obuku:

$
J = -\frac{1}{m} \sum_{j=1}^m \sum_{i=1}^n y_{ij} \, \log(a_{ij})
$


- Ako je $y^{(i)}=0$:  
  $\text{error} = -\log(1-a^{(i)})$  
  - Kada $a^{(i)} \to 0\ → greška ≈ 0$ (dobar pogodak).  
  - Kada $a^{(i)} \to 1\ → -\log(0) = +\infty\$ (veliki promašaj).  

- Ako je $y^{(i)}=1$:  
  $\text{error} = -\log(a^{(i)})$  
  - Kada $a^{(i)} \to 1 → greška ≈ 0$ (dobar pogodak).  
  - Kada $a^{(i)} \to 0 → \-\log(0) = +\infty\$ (veliki promašaj).  


---

### 🔹 Višeklasna klasifikacija

Ako imamo \(n\) klasa i koristimo one-hot kodiranje za oznake, funkcija troška glasi:

$
J = -\frac{1}{m} \sum_{j=1}^m \sum_{i=1}^n y_{ij} \, \log(a_{ij})
$


- $m$ = broj primera (observacija)
- $n$ = broj klasa (neurona u izlaznom sloju)  
- $y_{ij}$ = indikator (1 ako je primer \(j\) klase \(i\), 0 inače)  
- $a_{ij}$ = predviđena verovatnoća da primer \(j\) pripada klasi \(i\)




### Inicijalizacija i cilj
Na početku nasumično biramo parametre $W$ i $b$. Ovi parametri određuju vrednost loss funkcije, a cilj je **minimizacija** loss-a po parametrima. Da bismo to postigli, primenjujemo **Backpropagation** (BP).

U nastavku izvodimo BP za **binarnu klasifikaciju** sa **sigmoid** aktivacijom; zatim bismo mogli pokazati da se analogan rezultat dobija i u višeklasnom slučaju sa **softmax** funkcijom.

---

### Cost funkcija (binarna klasifikacija)

$
J = -\frac{1}{m}\sum_{i=1}^{m} \Big[y^{(i)}\log(a^{(i)}) + (1-y^{(i)})\log(1-a^{(i)})\Big]
$

---

### Arhitektura (2 skrivena sloja)

- $W^{[1]} \in \mathbb{R}^{n_1 \times n_0}$, $\quad$ $B^{[1]} \in \mathbb{R}^{n_1 \times 1}$  
- $W^{[2]} \in \mathbb{R}^{n_2 \times n_1}$, $\quad$ $B^{[2]} \in \mathbb{R}^{n_2 \times 1}$  
- $W^{[3]} \in \mathbb{R}^{n_3 \times n_2}$, $\quad$ $B^{[3]} \in \mathbb{R}^{n_3 \times 1}$  

Gde je:
- $m$ — broj observacija  
- $X \in \mathbb{R}^{n_0 \times m}$  
- $A^{[1]} \in \mathbb{R}^{n_1 \times m}$, $\quad$ $A^{[1]} = f_1(Z^{[1]})$  (npr. ReLU ili $\tanh$)  
- $A^{[2]} \in \mathbb{R}^{n_2 \times m}$, $\quad$ $A^{[2]} = f_2(Z^{[2]})$  
- $A^{[3]} \in \mathbb{R}^{n_3 \times m}$, $\quad$ $A^{[3]} = f_3(Z^{[3]})$  (za binarno: sigmoid)  
- $Y \in \mathbb{R}^{n_3 \times m}$

---

### Binarna klasifikacija i sigmoid

Cost (ponovo):

$
J = -\frac{1}{m}\sum_{i=1}^{m} \Big[y^{(i)}\log(a^{(i)}) + (1-y^{(i)})\log(1-a^{(i)})\Big]
$

Izlazni sloj:
$
Z^{[3]} = W^{[3]}A^{[2]} + B^{[3]}
$
$
\text{sigmoid}(x) = \frac{1}{1 + e^{-x}}
$

---

### Lančano pravilo za $W^{[3]}$

Želimo $\displaystyle \frac{\partial L}{\partial W^{[3]}}$. Koristimo:
$
\frac{\partial L}{\partial W^{[3]}}
;=\;
\frac{\partial L}{\partial A^{[3]}} \cdot
\frac{\partial A^{[3]}}{\partial Z^{[3]}} \cdot
\frac{\partial Z^{[3]}}{\partial W^{[3]}}.
$

1) Izvod po $A^{[3]}$:

$frac{\partial L}{\partial A^{[3]}}
= -\frac{y}{A^{[3]}} + \frac{1-y}{1-A^{[3]}}
= \frac{A^{[3]} - y}{A^{[3]}(1-A^{[3]})}.
$

2) Izvod sigmoide:
$
\frac{\partial A^{[3]}}{\partial Z^{[3]}}
= A^{[3]}\big(1-A^{[3]}\big).
$

3) Izvod linearnog dela:
$
\frac{\partial Z^{[3]}}{\partial W^{[3]}} = A^{[2]}.
$

Kombinovanjem:
$
\frac{\partial L}{\partial W^{[3]}}
= \left(\frac{A^{[3]} - y}{A^{[3]}(1-A^{[3]})}\right)
\cdot \big(A^{[3]}(1-A^{[3]})\big)
\cdot A^{[2]}
= (A^{[3]} - y)\,A^{[2]}.
$

### Izvodi za sloj 3

Analogno za $b^{[3]}$, koristeći $\frac{\partial z^{[3]}}{\partial b^{[3]}} = 1$:

$
\frac{\partial J}{\partial W^{[3]}} = \frac{1}{m}(A^{[3]} - Y)(A^{[2]})^{\top}
$

$
\frac{\partial J}{\partial b^{[3]}}
= \frac{\partial L}{\partial a^{[3]}} \cdot
  \frac{\partial a^{[3]}}{\partial z^{[3]}} \cdot
  \frac{\partial z^{[3]}}{\partial b^{[3]}}
$

$
\frac{\partial J}{\partial b^{[3]}}
= \frac{1}{m}\sum_{i=1}^m dZ^{[3]}_{(:,i)}
$

---

### Izvodi za sloj 2

Slično kao pre, sada pratimo lanac $L \to a^{[3]} \to z^{[3]} \to a^{[2]} \to z^{[2]} \to W^{[2]}$:

$
\frac{\partial L}{\partial W^{[2]}}
= \frac{\partial L}{\partial a^{[3]}}
\cdot \frac{\partial a^{[3]}}{\partial z^{[3]}}
\cdot \frac{\partial z^{[3]}}{\partial a^{[2]}}
\cdot \frac{\partial a^{[2]}}{\partial z^{[2]}}
\cdot \frac{\partial z^{[2]}}{\partial W^{[2]}}
$

- $\frac{\partial L}{\partial a^{[3]}} \cdot \frac{\partial a^{[3]}}{\partial z^{[3]}} = dZ^{[3]}$  
- $\frac{\partial z^{[3]}}{\partial a^{[2]}} = W^{[3]}$  
- $\frac{\partial a^{[2]}}{\partial z^{[2]}} = f_2'(Z^{[2]})$  
- $\frac{\partial z^{[2]}}{\partial W^{[2]}} = A^{[1]}$

Dakle:

$
\frac{\partial J}{\partial W^{[2]}}
= \frac{1}{m}\, dZ^{[2]} (A^{[1]})^{\top},
\quad dZ^{[2]} = (W^{[3]})^{\top}dZ^{[3]} \odot f_2'(Z^{[2]})
$

$
\frac{\partial J}{\partial b^{[2]}} = \frac{1}{m}\sum_{i=1}^m dZ^{[2]}_{(:,i)}
$

Dimenzije: $W^{[2]} \in \mathbb{R}^{n_2 \times n_1}$.

---

### Izvodi za sloj 1

Ostaje još sloj 1: lanac $L \to z^{[2]} \to a^{[1]} \to z^{[1]} \to W^{[1]}$.

$
\frac{\partial L}{\partial W^{[1]}}
= \frac{\partial z^{[2]}}{\partial W^{[1]}}
\cdot f_1'(Z^{[1]})
\cdot X
$

$
\frac{\partial J}{\partial W^{[1]}}
= \frac{1}{m}\, dZ^{[1]} X^{\top},
\quad dZ^{[1]} = (W^{[2]})^{\top}dZ^{[2]} \odot f_1'(Z^{[1]})
$

$
\frac{\partial J}{\partial b^{[1]}} = \frac{1}{m}\sum_{i=1}^m dZ^{[1]}_{(:,i)}
$


### Podaci i oblici
- Ulaz: slike 64×64×3 → vektor $n_0=12288$.
- Batched ulaz: $X\in\mathbb{R}^{n_0\times m}$ (kolone su primeri).
- Slojevi: $W^{[l]}\in\mathbb{R}^{n_l\times n_{l-1}},; b^{[l]}\in\mathbb{R}^{n_l\times 1}$.

### Forward
$Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]},\quad A^{[l]}=f^{[l]}(Z^{[l]})$, sa $A^{[0]}=X$.  
- Sigmoid: $\sigma(z)=1/(1+e^{-z})$, $\sigma'(z)=\sigma(1-\sigma)$.  
- ReLU: $f(z)=\max(0,z)$, $f'(z)=\mathbf{1}_{z>0}$.  
- Softmax: $a_i=\exp(z_i)/\sum_k \exp(z_k)$.

### Loss
- Binarno (sigmoid+BCE): $J=-\frac1m\sum_i \big[y^{(i)}\log a^{(i)}+(1-y^{(i)})\log(1-a^{(i)})\big]$.
- Multiklasno (softmax+CE): $J=-\frac1m\sum_{j=1}^m\sum_{i=1}^n y_{ij}\log a_{ij}$.

### Backprop (generalno)
- Izlaz:
  - BCE+sigmoid: $dZ^{[L]}=A^{[L]}-Y$.
  - CE+softmax: $dZ^{[L]}=A^{[L]}-Y$.
- Za $l=L,\dots,1$:
  - $dW^{[l]}=\frac1m\, dZ^{[l]}(A^{[l-1]})^\top$
  - $db^{[l]}=\frac1m\sum_{i=1}^m dZ^{[l]}_{(:,i)}$
  - $dA^{[l-1]}=(W^{[l]})^\top dZ^{[l]}$
  - $dZ^{[l-1]}=dA^{[l-1]}\odot f'^{[l-1]}(Z^{[l-1]})$

### Ažuriranje (SGD)
$W^{[l]}\leftarrow W^{[l]}-\eta\, dW^{[l]},\quad b^{[l]}\leftarrow b^{[l]}-\eta\, db^{[l]}$.

### Inicijalizacija
He (za ReLU): $W^{[l]}\sim \mathcal{N}\!\left(0,\frac{2}{n_{l-1}}\right),\; b^{[l]}=0$.



## Unapređivanje Neuronskih Mreža

U slučaju da su naše neuronske mreže **previše kompleksne** za skup podataka (npr. nemamo dovoljno primera ili koristimo “overkill” model), dolazi do **overfittinga**.  
Tada algoritam dobro radi na trening skupu (uči “napamet”), ali na novim podacima ne može da generalizuje.

---

### Najčešće tehnike za rešavanje:

- **Augmentacija slika**  
  Veštačko povećanje broja primera iz postojećih: sečenje, mirroring, rotiranje…  
  Za nas je to ista slika, ali za algoritam predstavlja novi uzorak sa drugačijom raspodelom piksela.

- **L2 regularizacija**  
  Kao i kod linearne regresije, uvodimo penalizaciju na težine \(W\).  
  Ovo sprečava da težine postanu prevelike i “previše zapamte” podatke.

- **Dropout**  
  Nasumično isključujemo određene neurone tokom treniranja.  
  Na taj način model ne može da se oslanja na jedan mali skup neurona, već mora da “raspodeli znanje”, što vodi ka boljoj generalizaciji.

---

### L2 Regularizacija: detaljno

Nova funkcija troška (sa regularizacijom):

$
J = \frac{1}{m} \sum_{i=1}^{m} L^{(i)}
+ \frac{\lambda}{2m}\sum_{l=1}^{L} \sum_{i} \sum_{j} \big(W_{ij}^{[l]}\big)^2
$

Primer širenja sume po težinama:

$
\sum W^2 = (W_{11}^{[1]})^2 + (W_{12}^{[1]})^2 + (W_{13}^{[1]})^2 + \dots +
(W_{21}^{[2]})^2 + (W_{22}^{[2]})^2 + \dots + (W_{ij}^{[l]})^2 + \dots
$

---

### Izvod u Backpropagation-u

Za deo regularizacije:

$
\frac{\partial}{\partial W_{ij}^{[l]}}
\left( \sum (W_{ij}^{[l]})^2 \right)
= 2W_{ij}^{[l]}.
$

Dakle, tokom backpropagation-a dobijamo dodatni član.  
Za gradient po težinama važi:

$
\frac{\partial J}{\partial W^{[l]}}
= \frac{1}{m}\, dZ^{[l]} (A^{[l-1]})^\top + \frac{\lambda}{m} W^{[l]}.
$

Za gradient po biasima nema dodatnog člana (regularizacija se ne primenjuje na $B$).

---


## Dropout

Pretpostavimo da u nekom sloju $l$ primenjujemo **Dropout**.  
Najčešće se primenjuje na nekoliko **poslednjih slojeva**, dok se **nikada ne stavlja na prvi (input) ni na poslednji (output)** sloj.  

---

### Forward Propagation

$
Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}
$
$
A^{[l]} = f(Z^{[l]})
$

Zatim kreiramo dropout masku:
$
D^{[l]} = (\text{np.random.rand}(A^{[l]}.shape) < \text{keep\_rate})
$

Primena maske:
$
A^{[l]} = A^{[l]} \odot D^{[l]}
$

---

#### Obrazloženje

- Cilj je da **nasumično isključimo određene neurone** iz sloja $A^{[l]}$.  
- Matrica $D^{[l]}$ ima iste dimenzije kao $A^{[l]}$.  
- U njoj se generišu vrednosti u intervalu [0,1], a poređenjem sa `keep_rate` dobijamo matricu sa **True/False** vrednostima.  
- Kada množimo $A^{[l]}$ sa $D^{[l]}$:  
  - ako je maska **True**, neuron ostaje aktivan,  
  - ako je maska **False**, vrednost neurona postaje 0 (isključen).  

Na ovaj način model tokom treniranja ne može da se oslanja na iste neurone svaki put, već uči robusnije reprezentacije.

---

### Backward Propagation

Za **poslednji sloj $L$** (gde se dropout ne primenjuje), računanje gradijenata ostaje standardno:

$
dZ^{[L]} = A^{[L]} - Y
$

$
dW^{[L]} = \frac{1}{m} dZ^{[L]} (A^{[L-1]})^\top
$

$
db^{[L]} = \frac{1}{m} \sum_{i=1}^m dZ^{[L]}_{(:,i)}
$

---

Za sve slojeve $l = 1,2,\dots,L-1$ gde je dropout primenjen, pravila se menjaju:

1. Propagacija unazad kroz težine:
$
dA^{[l]} = (W^{[l+1]})^\top dZ^{[l+1]}
$

2. Primena dropout maske:
$
dA^{[l]} = \frac{dA^{[l]} \odot D^{[l]}}{\text{keep\_rate}}
$

3. Aktivaciona funkcija:
$
dZ^{[l]} = dA^{[l]} \odot f'(Z^{[l]})
$

4. Gradijenti:

$
dW^{[l]} = \frac{1}{m} dZ^{[l]} (A^{[l-1]})^\top
$

$
db^{[l]} = \frac{1}{m} \sum_{i=1}^m dZ^{[l]}_{(:,i)}
$

---

### Napomena
- Faktor $\tfrac{1}{\text{keep\_rate}}$ se koristi da bismo **skalirali vrednosti** (inference mode → sve neurone koristimo, pa vrednosti ostaju očekivano iste).  
- Tokom testiranja/inference **dropout se isključuje**, ali se koristi skala težina koja obezbeđuje konzistentnost.


## Metode optimizacije cost funkcije i računanja hiperparametara W, B

Kao što smo rekli, parametri $W, B$ se računaju iterativnim ažuriranjem:

$
W \leftarrow W - \alpha \frac{\partial J}{\partial W},
\qquad
B \leftarrow B - \alpha \frac{\partial J}{\partial B},
$

gde je $\alpha$ **learning rate**.

---

###  Full Batch Gradient Descent

- **Ideja:** računa gradijent koristeći **ceo skup podataka**.  
- **Ažuriranje:**
$
W \leftarrow W - \alpha \frac{1}{m}\sum_{i=1}^m \frac{\partial L^{(i)}}{\partial W},
\qquad
B \leftarrow B - \alpha \frac{1}{m}\sum_{i=1}^m \frac{\partial L^{(i)}}{\partial B}.
$
- **Prednosti:** stabilna i brza konvergencija (linearno u okolini optimuma).  
- **Mane:** komputaciono vrlo zahtevno za velike skupove podataka.  

---

###  Stochastic Gradient Descent (SGD)

- **Ideja:** koristi **samo jedan podatak** u svakoj iteraciji.  
- **Ažuriranje:**
$
W \leftarrow W - \alpha \frac{\partial L^{(i)}}{\partial W},
\qquad
B \leftarrow B - \alpha \frac{\partial L^{(i)}}{\partial B}.
$
- **Prednosti:** jeftin po iteraciji, može brzo da “prošeta” parametre.  
- **Mane:** pravci spuštanja su bučni, konvergencija je sublinearna i oscilatorna.  
- **Napomena:** matematički se može pokazati da pod određenim uslovima dolazi do sublinearne konvergencije u okolini optimuma (dokazaćemo kasnije kroz naučne radove).  

---

### 🔹 Mini-Batch Gradient Descent

- **Ideja:** kompromis između **full batch** i **stochastic** pristupa.  
- Koristimo nasumično izabrani **mini-batch** $S$ veličine $m=|S|$ (tipično 32 ili 64).  
- **Ažuriranje:**
$
W \leftarrow W - \frac{\alpha}{m}\sum_{i\in S} \frac{\partial L^{(i)}}{\partial W},
\qquad
B \leftarrow B - \frac{\alpha}{m}\sum_{i\in S} \frac{\partial L^{(i)}}{\partial B}.
$
- **Prednosti:**  
  - Računski znatno efikasniji od full batch metode.  
  - Manje “bučan” od čistog SGD-a.  
  - Dobija se realnija aproksimacija pravog gradijenta.  
- **Mane:** kompromis – nije ni najbrže ni najstabilnije, ali u praksi se pokazalo kao najbolja opcija.  

---

### Konvergencija SGD algoritama kod neuronskih mreža

Bitno je napomenuti da radimo sa neuronskim mrežama (NNs) čije funkcije greške su **nelinearne** i **nekonveksne**.  
One čak **ne zadovoljavaju globalno PL (Polyak–Łojasiewicz) uslove**, pa stoga nije moguće pokazati **linearnu konvergenciju** prethodnih algoritama na celom skupu.  
Međutim, zadovoljavaju **PL i Ł uslove lokalno**.  

---

#### Zašto je to važno?

Pretpostavimo optimizaciju nelinearne nekonveksne funkcije.  
U praksi, **SGD često zalazi u oblasti gde je gradijent vrlo mali**, tj. u tzv. *ravne regione* optimizacione funkcije.  

- Znamo da Lipschitz uslov garantuje da se pravac gradijenta **ne menja naglo**,  
  ali to nije dovoljno jer:  
  1. Lipschitz gradijent **ne garantuje** da se male vrednosti gradijenta javljaju blizu rešenja.  
  2. SGD može da **luta ili stagnira** u zoni sa skoro nultim gradijentom, bez jasne konvergencije.  

Budući da SGD slučajno bira tačke i na osnovu njih određuje pravce,  
u “dolinama” koje sadrže beskonačno mnogo tačaka algoritam se ponaša kao da stalno traži neku slučajnu “optimalnu” tačku koja je najbliža trenutnoj iteraciji.  
Pošto se tačke biraju nasumično, i ta tačka optimuma kojoj algoritam prividno prilazi se stalno menja.  

---

#### Veza sa Łojasiewicz uslovom

Ako funkcija zadovoljava **Łojasiewicz-ev uslov**, tada važi relacija:

$
\|\nabla f(x)\| \;\geq\; L \, | f(x) - f(x^\star) |^\beta,
$

što znači da **mali gradijent implicira blizinu funkcionalne vrednosti optimuma**.  
To, pak, tera SGD da **konvergira ka konkretnoj tački**.  

Drugim rečima, ovde govorimo o **skoro sigurnoj konvergenciji**.

Rad iz koga sam izvukao ovaj zaključak-'Convergence of Stochastic Gradient Descent Schemes for  Lojasiewicz-Landscapes' - Steffen Dereich and Sebastian Kassing


# Konvolutivne Neuronske Mreže (CNN)

### Problem sa klasičnim neuronskim mrežama
Pretpostavimo zadatak klasifikacije slike dimenzija $1000 \times 1000 \times 3$  
(gde je 3 broj RGB kanala, a $1000 \times 1000\$ broj piksela po kanalu).

Ako bismo koristili potpuno povezane slojeve (fully connected), recimo da prvi sloj ima 1000 neurona:  
- Broj parametara bio bi:  
$
1000 \times (1000 \times 1000 \times 3) = 3 \cdot 10^9
$  
- To je **tri milijarde parametara samo za jedan sloj!**  
- Takva mreža bi bila:
  - računski neizvodljiva (memorijski i vremenski),
  - podložna **overfittingu** zbog ogromnog broja slobodnih parametara.

---

### Rešenje: Konvolucija
Ideja CNN-a:  
- Ne povezujemo svaki neuron sa svim pikselima ulaza.  
- Umesto toga, koristimo **filtere (jezgra, “kernels”)** koji imaju malu dimenziju (npr. $3 \times 3$, $5 \times 5$) i koji se “provlače” kroz sliku.  
- Svaki filter uči da prepoznaje određene **lokalne šare** (ivice, teksture, oblike).  
- Na ovaj način:
  - značajno se smanjuje broj parametara,
  - iskorišćava se lokalna struktura slike,
  - mreža postaje otpornija na **translacije** objekata (ako se objekat pomeri na slici, filter ga i dalje prepoznaje).

---

### Ključne osobine CNN-a
- **Lokalne konekcije**: neuron gleda samo “prozor” slike (npr. \(3 \times 3\) oblast).  
- **Deljenje težina**: isti filter se primenjuje na celoj slici → mnogo manje parametara.  
- **Hijerarhija osobina**:  
  - **Niži slojevi**: prepoznaju ivice, teksture.  
  - **Viši slojevi**: kombinuju ih u oblike, delove objekata i na kraju ceo objekat.  

---

### Intuicija konvolucije
Konvolucija se u najopštijem slučaju može posmatrati kao **mera sličnosti između dve funkcije**.  

U CNN-u naš filter (kernel) prolazi kroz sliku i računa koliko se lokalni deo slike poklapa sa obrascem koji filter “traži” — ivice, horizontalne/vertikalne linije, oblike itd.  

---

### Primer konvolucije

Pretpostavimo da imamo sliku formata \(6 \times 6 \times 1\), i filter dimenzija \(3 \times 3\):

$
\begin{bmatrix}
1 & 6 & 9 & 10 & 2 & 8 \\
2 & 5 & 1 & 8 & 4 & 2 \\
3 & 7 & 4 & 9 & 10 & 3 \\
9 & 8 & 3 & 6 & 7 & 9 \\
8 & 0 & 9 & 4 & 7 & 2 \\
9 & 10 & 12 & 6 & 9 & 8 \\
\end{bmatrix}
\quad * \quad
\begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1 \\
\end{bmatrix}
=
\begin{bmatrix}
-8 & -9 & -2 & 14 \\
6 & -3 & -13 & 9 \\
4 & -4 & -8 & 5 \\
2 & 2 & 1 & -3 \\
\end{bmatrix}
$

Na primer, poslednja vrednost $-3$ se računa kao:

$
6 \cdot 1 + 7 \cdot 0 + 9 \cdot (-1) +
4 \cdot 1 + 7 \cdot 0 + 2 \cdot (-1) +
6 \cdot 1 + 9 \cdot 0 + 8 \cdot (-1) = -3
$

Ovde pikseli imaju vrednosti od 0 do 255 (crno-bela slika).  

---

### Dimenzionalnost nakon konvolucije
- Za matricu $n \times n$ konvoluiranu filterom $f \times f$, rezultat ima dimenzije:  
  $
  (n-f+1) \times (n-f+1)
  $
- Ako koristimo $c$ filtera, dobijamo $c$ izlaznih slika (feature mapa):  
  $
  n \times n \times 1 * f \times f \times c
  \;\;\Rightarrow\;\;
  (n-f+1) \times (n-f+1) \times c
  $
- Za RGB sliku (3 kanala), filter mora imati dimenziju $f \times f \times 3$.  
  Kada koristimo $c$ takvih filtera, izlaz je:  
  $
  n \times n \times 3 * f \times f \times 3 \times c
  \;\;\Rightarrow\;\;
  (n-f+1) \times (n-f+1) \times c
  $

---

ℹ️ Ovo je osnova samo za **konvolucioni sloj**, bez **pooling-a** i **striding-a**.  
U nastavku se objašnjavaju pooling i stride — tehnike koje dodatno smanjuju dimenzionalnost i uvode translacionu invarijantnost.


### Ograničenja konvolucionih slojeva

Pored svih prednosti CNN-a, konvolucioni slojevi imaju i određene **mane**:

1. **Smanjenje informacija**  
   Sa svakim konvolucionim slojem dimenzije slike se smanjuju.  
   Ako bismo preterali sa primenom konvolucija bez drugih tehnika, izgubili bismo značajan deo informacija.

2. **Neravnomerna izloženost piksela filterima**  
   Posmatrajmo prethodni primer:  
   - piksel u gornjem levom uglu (vrednost **1**) izložen je filteru samo jednom.  
   - piksel u centru (npr. **4** iz trećeg reda i treće kolone) ulazi u mnogo više prozora filtera.  

   To znači da se **ivice slike obrađuju siromašnije** od centralnih delova.

---

### Rešenje: **Padding**

Da bi se ublažili ovi problemi koristi se **padding** — dodavanje sloja nula oko ivica slike pre konvolucije.  

Prednosti:
- Održava se dimenzija slike (tj. izlaz može imati iste dimenzije kao ulaz).  
- Sprečava preveliko smanjivanje slike kroz više slojeva.  
- Omogućava da i **ivice slike** budu više puta izložene filterima.  

Na primer, ako sliku $n \times n$ obložimo padding-om širine $p$, efektivna dimenzija postaje:

$
(n + 2p) \times (n + 2p)
$

Tada izlazna dimenzija nakon konvolucije sa filterom $f \times f$ postaje:

$
(n + 2p - f + 1) \times (n + 2p - f + 1)
$

Ako izaberemo $p = \frac{f-1}{2}$ (za neparne filtere), dobijamo tzv. **“same padding”** → izlaz ima iste dimenzije kao ulaz.


### Primer: Padding

Posmatrajmo malu sliku $4 \times 4$:

$
\begin{bmatrix}
1 & 2 & 3 & 4 \\
5 & 6 & 7 & 8 \\
9 & 10 & 11 & 12 \\
13 & 14 & 15 & 16 \\
\end{bmatrix}
$

---

#### Bez padding-a

Ako primenimo filter $3 \times 3$, izlazne dimenzije biće:

$
(4 - 3 + 1) \times (4 - 3 + 1) = 2 \times 2
$

Dakle, izlaz:

$
\begin{bmatrix}
* & * \\
* & * \\
\end{bmatrix}
$

Informacija se smanjila.

---

#### Sa padding-om \(p=1\)

Dodajemo sloj nula oko slike:

$
\begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 2 & 3 & 4 & 0 \\
0 & 5 & 6 & 7 & 8 & 0 \\
0 & 9 & 10 & 11 & 12 & 0 \\
0 & 13 & 14 & 15 & 16 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 \\
\end{bmatrix}
$

Dimenzije postaju $6 \times 6$.

Nakon konvolucije sa filterom $3 \times 3$:

$
(6 - 3 + 1) \times (6 - 3 + 1) = 4 \times 4
$

Rezultat sada ima iste dimenzije kao originalna slika $4 \times 4$.

---

Ovako **padding**:
- sprečava preveliko smanjenje dimenzija,  
- omogućava ivicama da budu “viđene” od strane filtera više puta,  
- čuva više informacija tokom obrade.


### Stride (korak pomeranja filtera)

Do sada smo prilikom pomeranja filtera išli **po jedan element** u desno ili dole.  
Ovo je u stvari **parametar koji biramo** i naziva se **stride** (korak).  

- Stride \(s = 1\): filter se pomera za jedan piksel (standardno).  
- Stride \(s = 2\): filter se pomera po dva piksela (preskače po jedan).  
- Veći stride → manji izlaz (feature map).

---

#### Primer

Ulazna matrica $5 \times 5$:

$
\begin{bmatrix}
1 & 4 & 6 & 7 & 8 \\
3 & 5 & 0 & -1 & 9 \\
3 & 7 & 3 & -3 & 7 \\
3 & 1 & 0 & -9 & -4 \\
2 & -3 & -1 & 5 & 3 \\
\end{bmatrix}
$

Filter $3 \times 3$:

$
\begin{bmatrix}
1 & 5 & 6 \\
3 & 2 & 5 \\
5 & 8 & 2 \\
\end{bmatrix}
$

Sa stride $s = 2$, dobijamo izlaz:

$
\begin{bmatrix}
a & b \\
c & d \\
\end{bmatrix}
$

---

#### Formula za dimenzije izlaza

Za ulaz dimenzija $n \times n$, filter dimenzija $f \times f$, stride $s$:

$
\text{output size} =
\Bigg(\left\lfloor \frac{n - f}{s} \right\rfloor + 1\Bigg)
\times
\Bigg(\left\lfloor \frac{n - f}{s} \right\rfloor + 1\Bigg)
$

Ako uključimo **padding** širine $p$:

$
\text{output size} =
\Bigg(\left\lfloor \frac{n + 2p - f}{s} \right\rfloor + 1\Bigg)
\times
\Bigg(\left\lfloor \frac{n + 2p - f}{s} \right\rfloor + 1\Bigg)
$

### Max Pooling

**Cilj**: smanjiti **komputacijski trošak** i dimenzije slike, a pritom očuvati što više informacija.  

**Ideja**:  
- Na izlazu iz konvolucionog sloja, podelimo sliku (feature map) u **prozore** dimenzija \(f \times f\).  
- U svakom prozoru zadržavamo **samo maksimalnu vrednost** (piksel sa najviše informacija).  
- Na taj način smanjujemo dimenzije, ali čuvamo ključne osobine.  

---

#### Primer

Ulazna matrica $4 \times 4$:

$
\begin{bmatrix}
1 & 3 & 2 & 4 \\
5 & 6 & 7 & 8 \\
2 & 4 & 9 & 1 \\
0 & 3 & 5 & 2 \\
\end{bmatrix}
$

Primena **Max Pooling-a** sa filterom $2 \times 2$, stride $s=2$:

- Prvi prozor: $\max(1,3,5,6) = 6$  
- Drugi prozor: $\max(2,4,7,8) = 8$  
- Treći prozor: $\max(2,4,0,3) = 4$  
- Četvrti prozor: $\max(9,1,5,2) = 9$  

Izlaz:

$
\begin{bmatrix}
6 & 8 \\
4 & 9 \\
\end{bmatrix}
$

---

#### Formula za dimenzije

Isto kao kod konvolucije:

$
\text{output size} =
\Bigg(\left\lfloor \frac{n + 2p - f}{s} \right\rfloor + 1\Bigg)
\times
\Bigg(\left\lfloor \frac{n + 2p - f}{s} \right\rfloor + 1\Bigg)
$

gde su:
- $n$ — ulazna dimenzija (visina/širina),  
- $f$ — veličina pooling filtera,  
- $s$ — stride,  
- $p$ — padding (retko se koristi kod poolinga).


### Povezivanje CNN-a i klasične FNN (Fully Connected) mreže

Do sada smo gradili konvolucione slojeve (Conv, ReLU, Pool).  
Na poslednjem konvolucionom sloju dobijamo objekat dimenzija:

$
H_{\text{out}} \times W_{\text{out}} \times m
$

gde je:
- $H_{\text{out}}$ — visina izlaza,
- $W_{\text{out}}$ — širina izlaza,
- $m$ — broj filtera korišćenih u poslednjoj konvoluciji.

---

#### Flatten sloj

Da bismo mogli da predviđamo **kategorije** i definišemo **loss funkciju** za učenje, ovaj tenzor moramo izravnati u **vektor**:

$
(H_{\text{out}} \cdot W_{\text{out}} \cdot m, \; 1)
$

Ovaj vektor postaje ulaz u **klasičnu potpuno povezanu neuronsku mrežu (FNN)**.

---

#### Fully Connected slojevi (klasični deo)

U ovom delu mreže ponovo koristimo standardnu formu:

$
Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]},
\quad
A^{[l]} = f^{[l]}(Z^{[l]}),
\quad
A^{[0]} = X
$

**Bitna napomena:**  
U formuli za FNN slojeve simbol $W^{[l]}$ označava **parametre koje optimizujemo** (matrice težina).  
Međutim, u CNN kontekstu $W^{[l]}$ obično označava **širinu tenzora**, dok se parametri sloja (kerneli) označavaju sa $K^{[l]}$.  
Da bismo izbegli konfuziju:  
- u FNN-u koristimo $W^{[l]}$ za težine,  
- u CNN-u koristimo $K^{[l]}$ za kernele, a $H^{[l]}, W^{[l]}, C^{[l]}$ za dimenzije tenzora.

gde je $X$ naš **flatten-ovan vektor** dimenzija $
(H_{\text{out}} \cdot W_{\text{out}} \cdot m, \; 1)
$.

---

#### Aktivacione funkcije

- **Sigmoid**:  
  $
  \sigma(z) = \frac{1}{1+e^{-z}},
  \quad
  \sigma'(z) = \sigma(z)(1-\sigma(z))
  $

- **ReLU**:  
  $
  f(z) = \max(0,z),
  \quad
  f'(z) = \mathbf{1}_{z>0}
  $

- **Softmax** (izlazni sloj kod multiklasne klasifikacije):  
  $
  a_i = \frac{\exp(z_i)}{\sum_k \exp(z_k)}
  $


### Forward propagacija u CNN sloju

Za $l$-ti konvolucioni sloj izlaz se računa kao:

$
Z^{[l]}[i,j,k]
= \sum_{u=0}^{f^{[l]}-1} \sum_{v=0}^{f^{[l]}-1} \sum_{c=0}^{C^{[l-1]}-1}
A^{[l-1]}[\,i+u, \; j+v, \; c\,] \cdot K^{[l]}[\,u,v,c,k\,] \; + \; b^{[l]}_k
$

gde je:
- $Z^{[l]}[i,j,k]$ — vrednost izlazne mape osobina (feature map) u tački $(i,j)$ za filter $k$.
- $f^{[l]}$ — dimenzija filtera (širina/visina).
- $C^{[l-1]}$ — broj kanala u ulazu (dubina prethodnog sloja).
- $K^{[l]}$ — skup filtera (tenzor dimenzija $f^{[l]} \times f^{[l]} \times C^{[l-1]} \times m^{[l]}$).
- $b^{[l]}_k$ — bias koji pripada filteru $k$.

---

### Dimenzije izlaza

Ako je ulazna dimenzija sloja $(H^{[l-1]}, W^{[l-1]}, C^{[l-1]})$, tada važi:

$
H^{[l]} = \left\lfloor \frac{H^{[l-1]} + 2p^{[l]} - f^{[l]}}{s^{[l]}} \right\rfloor + 1,
$

$
W^{[l]} = \left\lfloor \frac{W^{[l-1]} + 2p^{[l]} - f^{[l]}}{s^{[l]}} \right\rfloor + 1,
$

$
m^{[l]} = \text{broj filtera u sloju } l.
$

Dakle, izlaz je tenzor:

$
Z^{[l]} \in \mathbb{R}^{H^{[l]} \times W^{[l]} \times m^{[l]}}.
$

---

### Važna napomena
- Svaki filter u sloju $l$ **ima dubinu jednaku broju filtera prethodnog sloja**:  
  $
  C^{[l]} = m^{[l-1]}.
  $
- Na taj način filter “vidi” sve kanale prethodnog sloja, a rezultat je jedna izlazna mapa osobina.

Na dobijeni tenzor $Z^{[l]}$ primenjujemo odgovarajuću aktivacionu funkciju
(npr. ReLU ili Tanh).  
Kada dođemo do poslednjeg konvolucionog sloja, nakon primene aktivacione funkcije
rezultat **flatten-ujemo** u vektor (kao što je ranije opisano).  
Ovaj vektor zatim prosleđujemo u već opisani **Forward Propagation kroz FNN slojeve**.



## 🔹 Backpropagacija kroz konvolucioni sloj

### 1. Chain rule za kernel
Za parametre kernela važi:
$
\frac{\partial L}{\partial K^{[l]}}
= \frac{\partial L}{\partial Z^{[l]}} \cdot \frac{\partial Z^{[l]}}{\partial K^{[l]}}
$

Gde:
$
\frac{\partial L}{\partial Z^{[l]}}
= \frac{\partial L}{\partial A^{[l]}} \odot g'^{[l]}(Z^{[l]}),
$
a $g^{[l]}$ je aktivaciona funkcija u sloju \(l\).

---

### 2. Forward relacija
Iz forward propagacije:
$
Z^{[l]}[i,j,k]
= \sum_{u=0}^{f-1} \sum_{v=0}^{f-1} \sum_{c=0}^{C^{[l-1]}-1}
A^{[l-1]}[i+u, j+v, c] \cdot K^{[l]}[u,v,c,k] + b^{[l]}_k
$

---

### 3. Parcijalni izvod \(Z\) po kernelu
Za jednu težinu $K^{[l]}[u,v,c,k']$:
$
\frac{\partial Z^{[l]}[i,j,k]}{\partial K^{[l]}[u,v,c,k']} =
\begin{cases}
A^{[l-1]}[i+u, j+v, c], & \text{ako } k' = k, \\
0, & \text{ako } k' \neq k.
\end{cases}
$

---

### 4. Intuicija
- Svaki filter $k$ generiše **samo svoj izlazni kanal** $Z[:,:,k]$.  
- Težine kernela $k$ utiču **isključivo** na odgovarajući kanal.  
- Dakle:
  - ako je $k'=k$ → izvod je jednak vrednosti lokalnog ulaza $A^{[l-1]}[i+u,j+v,c]$,  
  - ako je $k' \neq k$ → izvod je $0$, jer kernel ne učestvuje u računanju tog kanala.

---
## Dalji racun

Želimo da izračunamo:

$
\frac{\partial L}{\partial K^{[l]}}
= A^{[l-1]} * \frac{\partial Z^{[l]}}{\partial K^{[l]}}
$

ili, sa mini-batchom ($m$ primera):

$
\frac{\partial L}{\partial K^{[l]}}
= \sum_{i=1}^{m} \text{conv}\big(A^{[l-1](i)}, \; \partial Z^{[l](i)}\big)
$

---

### Gradijent po ulazu
Preostaje nam još da nađemo:

$
\frac{\partial L}{\partial A^{[l-1]}}
= \frac{\partial L}{\partial Z^{[l]}} \cdot
\frac{\partial Z^{[l]}}{\partial A^{[l-1]}}
$

gde je:

$
\frac{\partial L}{\partial Z^{[l]}}
= \partial A^{[l]} \cdot g'^{[l]}(Z^{[l]})
$

$
Z^{[l]} = A^{[l-1]} * K^{[l]} + b^{[l]}
$

---

### Transponovana konvolucija
Ne postoji “inverzna konvolucija”.  
U backpropagaciji CNN-a koristimo tzv. **transponovanu konvoluciju**.  

Umesto da računamo $\text{conv}(A, K)$, računamo:

$
\frac{\partial L}{\partial A^{[l-1]}}
= \text{conv}\Big(\frac{\partial L}{\partial Z^{[l]}}, \; \text{flip}(K^{[l]})\Big)
$

gde **flip** znači rotiranje kernela za $180^\circ$.

---

### Početni uslov
Takođe, kao što se može primetiti, $\frac{\partial L}{\partial Z^{[l]}}$ se dobija preko $\partial A^{[l]}$.  
Međutim, početni $\partial A^{[L]}$ nam je poznat i on dolazi direktno **prebacivanjem iz FNN dela** nazad u matricni oblik dimenzija $H^{[l]} \times W^{[l]} \times m^{[l]}$.  
Dakle, sve je prisutno za izvođenje backpropagacije i dobijanje optimalnih parametara $K$.

---

### Gradijent po biasu
Za bias važi:

$
\frac{\partial L}{\partial b^{[l]}}
= \sum_{i,j,k} \frac{\partial L}{\partial Z^{[l]}_{i,j,k}} \cdot
\frac{\partial Z^{[l]}_{i,j,k}}{\partial b^{[l]}}
$

Drugi deo je jednak $1$, pa dobijamo:

$
\frac{\partial L}{\partial b^{[l]}} = \sum_{i,j,k} \frac{\partial L}{\partial Z^{[l]}_{i,j,k}}
$









