#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
75 : ce que les SEQUENCES ordonnees ajoutent aux paires, et ce qu'elles ne peuvent pas ajouter.

QUESTION A L'ORIGINE (Caroline Hillairet, compte rendu du 13 aout 2026) : exploiter les
probabilites CONDITIONNELLES de sequence, du type P1 -> P2 -> P3 contre P2 -> P3 -> P1, pour
reconstituer la composante dirigee A que la co-occurrence seule n'identifie pas.

L'idee est juste dans son principe : des marges par paires ne determinent pas une loi sur les
permutations, donc un triplet porte en general une information qu'aucune paire ne porte. Ce script
va la chercher dans le corpus, et le resultat est negatif de trois facons differentes qu'il faut
distinguer, parce qu'elles ne se corrigent pas de la meme maniere.

CE QUE LE SCRIPT ETABLIT, DANS L'ORDRE :
  1. les sequences sont DEJA dans le corpus, implicites : chaque incident code un ensemble
     d'aretes dirigees, et composer ces aretes donne des chemins. Personne ne les avait extraits ;
  2. la question posee ne peut pas etre tranchee, et pour une raison structurelle : P2 n'EMET
     jamais, dans 22 transitions sur 22. Les deux sequences comparees exigent toutes deux que P2
     emette, donc aucune des deux n'a le moindre support ;
  3. le graphe agrege est ACYCLIQUE, d'ordre topologique P1 < {P4,P5} < P3 < P2. On mesure si
     c'est surprenant, sous deux lois nulles dont une seule est legitime, et la reponse est non ;
  4. le modele, lui, N'EST PAS sans memoire sur les sequences : l'auto-evitement fait dependre le
     successeur du predecesseur, avec un facteur de gonflement calculable. Les triplets portent
     donc une prediction testable, ce qui contredit l'idee qu'ils seraient redondants ;
  5. mais le test n'a AUCUNE puissance sur ce corpus, et pas faute d'echantillon. Il exige un
     pilier a la fois ATTEINT (sinon pas de predecesseur) et a DEUX successeurs distincts (sinon
     la loi conditionnelle egale la marginale). Les deux manques sont EXCLUSIFS ici : P1 a trois
     successeurs mais n'est jamais atteint, P3/P4/P5 sont atteints mais n'ont qu'un successeur.
     Un corpus dix fois plus grand du meme genre n'y changerait rien.

CONSEQUENCE POUR L'IDENTIFICATION : les sequences ne resserrent pas la bande. Ce n'est pas un
renoncement mais un diagnostic, et il dit ce qu'il faudrait observer pour que cela change.

Sortie : diagnostics + figure S31_sequences_ordonnees.png.
"""

import itertools
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import scr_engine as eng                     # noqa: E402
import postmortem_corpus as pc               # noqa: E402

WID = 88
PIL = eng.PIL
TRANS = eng.TRANS
SROW = eng._SROW
MAXS = eng._MAXS
G_NC = 0.90                                  # gain de l'etat non conforme
NOM = {1: "P1 gouvernance", 2: "P2 reponse", 3: "P3 detection", 4: "P4 tiers",
       5: "P5 partage"}
SEED = 20260813


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


CORPUS = pc.CORPUS
ARETES = [(c["nom"], a, b) for c in CORPUS for a, b, _ in c["transitions"]]

# =====================================================================================
titre("1. Les sequences sont deja dans le corpus, il suffisait de composer les aretes")
# =====================================================================================
# Un incident code un ENSEMBLE d'aretes dirigees. Un chemin de longueur 2 est une paire
# d'aretes j->k et k->l du MEME incident, avec l different de j. C'est une sequence observee.
triplets = Counter()
par_incident = {}
for c in CORPUS:
    e = [(a, b) for a, b, _ in c["transitions"]]
    tri = [(a, b, d) for (a, b) in e for (x, d) in e if x == b and d != a]
    par_incident[c["nom"]] = tri
    triplets.update(tri)

n_tri = sum(triplets.values())
print(f"  {len(CORPUS)} incidents, {len(ARETES)} transitions codees.")
print(f"  Chemins de longueur 2 obtenus par composition : {n_tri}, "
      f"pour {len(triplets)} sequences distinctes.")
print(f"\n  {'sequence':<22}{'occurrences':>13}   incidents")
for (a, b, d), n in sorted(triplets.items(), key=lambda kv: (-kv[1], kv[0])):
    qui = [nom.split()[0] for nom, t in par_incident.items() if (a, b, d) in t]
    print(f"  P{a} -> P{b} -> P{d}{'':<9}{n:>13}   {', '.join(qui)}")
sans = [nom for nom, t in par_incident.items() if not t]
print(f"\n  {len(sans)} incidents ne fournissent AUCUN chemin de longueur 2 : "
      f"{', '.join(n.split()[0] for n in sans)}.")
print("  Soit ils ne codent qu'une arete, soit leurs aretes partent toutes du meme pilier.")

# chaine la plus longue observee
def chemins_longs(e, k):
    """Chemins simples de longueur k aretes dans l'ensemble d'aretes e."""
    out = []
    for depart in {a for a, _ in e}:
        pile = [(depart,)]
        while pile:
            ch = pile.pop()
            if len(ch) == k + 1:
                out.append(ch)
                continue
            for (a, b) in e:
                if a == ch[-1] and b not in ch:
                    pile.append(ch + (b,))
    return out


for k in (3, 4):
    tot = []
    for c in CORPUS:
        e = [(a, b) for a, b, _ in c["transitions"]]
        for ch in chemins_longs(e, k):
            tot.append((c["nom"], ch))
    if tot:
        print(f"\n  Chemins de longueur {k} aretes : {len(tot)}")
        for nom, ch in tot:
            print(f"    {' -> '.join('P' + str(x) for x in ch)}   ({nom})")
    else:
        print(f"\n  Aucun chemin de longueur {k} aretes.")

# =====================================================================================
titre("2. La question posee ne peut pas etre tranchee, et la raison est structurelle")
# =====================================================================================
emet = Counter(a for _, a, _ in ARETES)
recoit = Counter(b for _, _, b in ARETES)
print(f"  {'pilier':<18}{'emissions':>12}{'receptions':>12}   role dans le corpus")
for j in PIL:
    if emet[j] and not recoit[j]:
        role = "SOURCE PURE"
    elif recoit[j] and not emet[j]:
        role = "PUITS PUR"
    elif emet[j] and recoit[j]:
        role = "intermediaire"
    else:
        role = "absent"
    print(f"  {NOM[j]:<18}{emet[j]:>12}{recoit[j]:>12}   {role}")

print(f"\n  P2 n'emet JAMAIS : {emet[2]} emission sur {len(ARETES)} transitions.")
print(f"  P1 n'est JAMAIS cible : {recoit[1]} reception sur {len(ARETES)}.")
print("\n  Or les deux sequences que la question compare exigent toutes deux que P2 emette :")
print("    P1 -> P2 -> P3  demande l'arete P1->P2, jamais codee, puis P2->P3, jamais codee ;")
print("    P2 -> P3 -> P1  demande P2->P3, jamais codee, puis P3->P1, impossible P1 etant")
print("                    source pure.")
print("  Aucune des deux n'a le moindre support. La comparaison n'est donc pas indecise faute")
print("  de donnees, elle est SANS OBJET sur ce corpus : ce n'est pas un intervalle large, c'est")
print("  un ensemble vide. La reponse a donner est de corriger la question.")
print("\n  ET IL FAUT DISTINGUER DEUX RELATIONS QUE LE MOT « ordre » CONFOND. Le corpus code le")
print("  CONDITIONNEMENT DIRECT (le rapport etablit que j a precede ET conditionne k), pas la")
print("  PRECEDENCE (j est arrive avant k). La cloture transitive des aretes le montre :")
directe = {(a, b) for _, a, b in ARETES}
clot = set(directe)
for _ in range(len(PIL)):
    ajout = {(a, d) for (a, b) in clot for (x, d) in clot if x == b and a != d}
    if ajout <= clot:
        break
    clot |= ajout
nouvelles = sorted(clot - directe)
print(f"    aretes codees en direct        : {len(directe)}  {sorted(directe)}")
print(f"    ajoutees par transitivite      : {len(nouvelles)}  {nouvelles}")
# combien d'incidents IMPLIQUENT chaque paire ajoutee, sans la coder
for (a, d) in nouvelles:
    n_imp = sum(1 for c in CORPUS
                if any(t[0] == a and t[2] == d for t in par_incident[c["nom"]]))
    print(f"    la paire (P{a},P{d}) est impliquee par {n_imp} incident"
          f"{'s' if n_imp > 1 else ''} et codee par aucun")
print("  Donc P1 precede P2 sans jamais le conditionner DIRECTEMENT. Pour W c'est correct, W")
print("  etant un transfert direct ; mais lire p_jk comme une precedence serait faux, et le")
print("  p_12 = 0 du script 59 ne dit pas que P2 pourrait preceder P1.")

# =====================================================================================
titre("3. Le graphe agrege est acyclique, et ce n'est PAS surprenant")
# =====================================================================================
paires = sorted({tuple(sorted((a, b))) for _, a, b in ARETES})
mult = Counter(tuple(sorted((a, b))) for _, a, b in ARETES)
print(f"  {len(ARETES)} transitions se repartissent sur {len(paires)} paires distinctes :")
for p in paires:
    d = [(a, b) for _, a, b in ARETES if tuple(sorted((a, b))) == p][0]
    print(f"    {{P{p[0]},P{p[1]}}}  multiplicite {mult[p]:>2}, "
          f"toujours codee P{d[0]} -> P{d[1]}")


def acyclique(aretes):
    """Vrai si le graphe dirige est sans cycle (tri topologique de Kahn)."""
    n = defaultdict(int)
    adj = defaultdict(list)
    noeuds = {x for e in aretes for x in e}
    for a, b in aretes:
        adj[a].append(b)
        n[b] += 1
    file = [x for x in noeuds if n[x] == 0]
    vus = 0
    while file:
        x = file.pop()
        vus += 1
        for y in adj[x]:
            n[y] -= 1
            if n[y] == 0:
                file.append(y)
    return vus == len(noeuds)


obs = sorted(directe)
print(f"\n  Le graphe agrege est acyclique : {acyclique(obs)}.")
ordre = []
restant = {x for e in obs for x in e}
ar = list(obs)
while restant:
    libres = sorted(x for x in restant if not any(b == x for a, b in ar if a in restant))
    ordre.append(libres)
    restant -= set(libres)
    ar = [(a, b) for a, b in ar if a in restant and b in restant]
print("  Ordre topologique : " + " < ".join("{" + ",".join("P" + str(x) for x in n) + "}"
                                            for n in ordre))
# COMBIEN L'ORDRE EST-IL CONTRAINT ? La cloture transitive tranche certaines paires et pas
# d'autres. C'est la mesure de ce que le corpus dit reellement de l'ordre.
toutes = list(itertools.combinations(PIL, 2))
tranchees = [p for p in toutes if (p[0], p[1]) in clot or (p[1], p[0]) in clot]
incomp = [p for p in toutes if p not in tranchees]
print(f"  La cloture transitive tranche {len(tranchees)} des {len(toutes)} paires de piliers ; "
      f"restent incomparables :")
print("    " + ", ".join(f"(P{a},P{b})" for a, b in incomp))
print("  L'ordre est donc PRESQUE total, et les deux paires libres sont exactement celles qui")
print("  n'ont jamais ete vues ensemble dans une meme chaine.")

# NUL 1, LEGITIME : chaque PAIRE distincte recoit une direction au hasard. C'est le nul qui
# correspond a la question « existe-t-il une direction », une paire etant une relation.
tot = 0
acy = 0
for bits in itertools.product([0, 1], repeat=len(paires)):
    g = [(p[0], p[1]) if b == 0 else (p[1], p[0]) for p, b in zip(paires, bits)]
    tot += 1
    acy += acyclique(g)
print(f"\n  NUL PAR PAIRE ({tot} orientations equiprobables des {len(paires)} paires) :")
print(f"    {acy} sont acycliques, soit {100*acy/tot:.1f} %. L'acyclicite observee n'est donc")
print(f"    PAS un resultat : p = {acy/tot:.2f}, on l'obtiendrait deux fois sur trois au hasard.")
print("    Un graphe de six aretes sur cinq sommets est trop peu contraint pour que son")
print("    acyclicite dise quoi que ce soit.")

# NUL 2, ILLEGITIME MAIS INSTRUCTIF : chaque OBSERVATION change de sens independamment. Il rend
# l'acyclicite ecrasante, mais il teste la constance du codeur et non l'existence d'une direction.
rng = np.random.default_rng(SEED)
NSIM = 20000
k = 0
for _ in range(NSIM):
    g = [(a, b) if rng.random() < 0.5 else (b, a) for _, a, b in ARETES]
    k += acyclique(g)
print(f"\n  NUL PAR OBSERVATION ({NSIM} tirages, chaque transition changeant de sens a 1/2) :")
print(f"    {100*k/NSIM:.2f} % sont acycliques. L'acyclicite paraitrait ecrasante, et c'est")
print("    trompeur : ce nul detruit la CONSTANCE du codage, une paire vue cinq fois pouvant")
print("    recevoir cinq directions differentes. Il mesure la fiabilite du codeur, pas")
print("    l'existence d'une direction. C'est le meme travers que l'indexation sur la valeur")
print("    dans le harnais : le nul doit porter sur l'objet, ici la relation, pas sur ses")
print("    occurrences.")

# =====================================================================================
titre("4. Le modele n'est PAS sans memoire sur les sequences, et c'est calculable")
# =====================================================================================
print("  On lit souvent que la chaine des piliers est sans memoire, donc que les triplets")
print("  seraient redondants avec les paires. C'est faux pour CE modele, et pour une raison")
print("  qui n'est pas un detail : la cascade est AUTO-EVITANTE. Depuis k, la cible est tiree")
print("  dans TRANS[k]/s_k, et un pilier deja tombe eteint la propagation. Donc, conditionnellement")
print("  a ce que la cascade CONTINUE vers un pilier neuf, la loi du successeur est")
print("  TRANS[k][l] / (s_k - somme des TRANS[k] vers les piliers deja visites) : elle depend du")
print("  chemin parcouru, donc du predecesseur.")
print(f"\n  {'triplet':<20}{'sans memoire':>14}{'auto-evitant':>14}{'gonflement':>12}")
gonf = []
for (a, b, d), _ in sorted(triplets.items(), key=lambda kv: (-kv[1], kv[0])):
    p_naif = TRANS[b][d] / SROW[b]
    dispo = SROW[b] - sum(w for kk, w in TRANS[b].items() if kk in (a, b))
    p_evit = TRANS[b][d] / dispo if dispo > 0 else float("nan")
    gonf.append((f"P{a}->P{b}->P{d}", p_naif, p_evit, p_evit / p_naif))
    print(f"  P{a} -> P{b} -> P{d}{'':<7}{p_naif:>14.4f}{p_evit:>14.4f}"
          f"{p_evit/p_naif:>11.3f}x")
g_moy = float(np.mean([g[3] for g in gonf]))
print(f"\n  Gonflement moyen {g_moy:.3f}x, maximum {max(g[3] for g in gonf):.3f}x. Le modele")
print("  predit donc des sequences MESURABLEMENT differentes de ce qu'une chaine sans memoire")
print("  donnerait, et les triplets ne sont pas redondants : ils portent une prediction.")
print("\n  UNE PRECISION SUR CE CALCUL, ET ELLE VA DANS LE SENS FAVORABLE. Le gonflement est")
print("  calcule pour un chemin qui DEMARRE au premier pilier du triplet, donc avec deux piliers")
print("  seulement dans l'ensemble visite. Si le triplet est lui-meme precede, comme le")
print("  P5 -> P3 -> P2 d'Equifax qui suit un P1 -> P5, l'exclusion porte sur trois piliers et le")
print("  gonflement est plus grand. Les valeurs ci-dessus sont donc des BORNES BASSES de l'ecart")
print("  a l'absence de memoire.")

# loi exacte des sequences sous le modele, jamais publiee
def loi_sequences(g):
    """Loi EXACTE des chemins de la cascade auto-evitante, ponderee par la propension d'amorce."""
    w = np.array([eng.LAMBDA[j] for j in PIL], float)
    w = w / w.sum()
    out = Counter()

    def rec(cur, ch, p):
        e = g * SROW[cur] / MAXS
        srow = SROW[cur]
        p_stop = 1.0 - e + e * sum(x for kk, x in TRANS[cur].items() if kk in ch) / srow
        out[ch] += p * p_stop
        for kk, x in TRANS[cur].items():
            if kk not in ch:
                rec(kk, ch + (kk,), p * e * x / srow)

    for c, j in enumerate(PIL):
        rec(j, (j,), w[c])
    return out


loi = loi_sequences(G_NC)
tot_p = sum(loi.values())
print(f"\n  LOI EXACTE DES SEQUENCES SOUS LE MODELE (g = {G_NC}, somme = {tot_p:.6f}) :")
print(f"  {'sequence':<24}{'probabilite':>13}   rang")
top = sorted(loi.items(), key=lambda kv: -kv[1])[:10]
for r, (ch, p) in enumerate(top, 1):
    print(f"  {' -> '.join('P' + str(x) for x in ch):<24}{100*p:>12.3f} %{r:>7}")
long2 = {ch: p for ch, p in loi.items() if len(ch) == 3}
s2 = sum(long2.values())
print(f"\n  Les chemins de longueur 2 aretes pesent {100*s2:.2f} % de la masse.")
print("  Classement du modele sur ces chemins, contre le comptage du corpus :")
print(f"  {'sequence':<20}{'modele':>10}{'rang mod.':>11}{'corpus':>9}{'rang corp.':>12}")
mod2 = sorted(long2.items(), key=lambda kv: -kv[1])
rang_mod = {ch: r for r, (ch, _) in enumerate(mod2, 1)}
obs2 = sorted(triplets.items(), key=lambda kv: -kv[1])
for r, ((a, b, d), n) in enumerate(obs2, 1):
    ch = (a, b, d)
    print(f"  P{a}->P{b}->P{d}{'':<11}{100*long2.get(ch, 0):>9.3f}%{rang_mod.get(ch, 0):>11}"
          f"{n:>9}{r:>12}")
print(f"\n  CE RAPPROCHEMENT N'EST PAS UN TEST, ET IL NE FAUT PAS LE PRESENTER COMME TEL. La loi")
print(f"  du modele est INCONDITIONNELLE sur {len(mod2)} chemins de longueur 2, alors que le corpus")
print("  n'est pas un echantillon tire au hasard : ce sont des post-mortems d'incidents notables,")
print("  choisis parce qu'ils ont fait l'objet d'un rapport. Le biais de selection est celui que")
print("  le script 64 mesure, et il interdit de lire les frequences du corpus comme des")
print("  probabilites. Ce qu'on peut dire est plus modeste : les deux sequences les plus")
print(f"  frequentes du corpus sont aux rangs {rang_mod.get(obs2[0][0])} et "
      f"{rang_mod.get(obs2[1][0])} du modele sur {len(mod2)}, donc il n'y a pas de")
print("  contradiction criante, et rien de plus.")

# =====================================================================================
titre("5. Le test a une puissance NULLE, et pas faute d'echantillon")
# =====================================================================================
succ = defaultdict(Counter)
for _, a, b in ARETES:
    succ[a][b] += 1
print("  UN TEST DE DEPENDANCE AU PREDECESSEUR EXIGE DEUX CHOSES DU MEME PILIER : qu'il soit")
print("  ATTEINT, sans quoi il n'y a pas de predecesseur sur quoi conditionner, et qu'il ait au")
print("  moins DEUX successeurs distincts, sans quoi la loi conditionnelle egale la marginale")
print("  par construction. Voici les deux conditions, pilier par pilier.")
print(f"\n  {'pilier':<16}{'atteint':>9}{'emis':>7}{'cibles':>8}   test possible ?")
testables = []
for j in PIL:
    d = len(succ[j])
    ok = recoit[j] >= 1 and d >= 2
    if ok:
        testables.append(j)
    detail = ", ".join(f"P{k}:{v}" for k, v in sorted(succ[j].items())) or "aucune"
    diag = ("OUI" if ok else
            ("non, jamais atteint" if recoit[j] == 0 else
             ("non, loi degeneree" if d == 1 else "non, n'emet pas")))
    print(f"  {NOM[j]:<16}{recoit[j]:>9}{emet[j]:>7}{d:>8}   {diag}   ({detail})")

print(f"\n  AUCUN PILIER NE REMPLIT LES DEUX CONDITIONS : {len(testables)} sur {len(PIL)}. Et les deux")
print("  manques sont exclusifs l'un de l'autre dans ce corpus, ce qui est plus net qu'une")
print("  simple penurie. P1 a bien trois successeurs distincts, donc une loi non degeneree,")
print("  mais il n'est JAMAIS atteint : il n'y a pas de predecesseur a conditionner. P3, P4 et")
print("  P5 sont atteints, mais leur loi de successeur est un POINT : depuis P3 tout va a P2,")
print("  depuis P4 tout va a P2, depuis P5 tout va a P3. La statistique du test est donc")
print("  identiquement nulle, et le test n'a pas une faible puissance, il n'en a AUCUNE.")
print("\n  CE N'EST PAS UN PROBLEME DE TAILLE D'ECHANTILLON, et c'est ce qui rend le diagnostic")
print("  utile plutot que decevant. Ajouter dix incidents du meme genre laisserait les lois")
print("  degenerees et le test sans prise. Le critere de reouverture est donc precis : il faut")
print("  un pilier a la fois ATTEINT et observe avec DEUX successeurs distincts, repete assez")
print("  souvent. C'est une condition sur la STRUCTURE du corpus, pas sur son volume.")
# combien faudrait-il ? loi binomiale exacte : n triplets par le meme pilier, deux successeurs
from math import comb                                                        # noqa: E402


def n_min(p0, p1, alpha=0.05, puissance=0.80):
    """Plus petit n tel qu'un test binomial exact separe p0 de p1 (unilateral)."""
    for n in range(2, 600):
        seuil = None
        for k in range(n, -1, -1):
            q = sum(comb(n, i) * p0 ** i * (1 - p0) ** (n - i) for i in range(k, n + 1))
            if q > alpha:
                seuil = k + 1
                break
        if seuil is None or seuil > n:
            continue
        pw = sum(comb(n, i) * p1 ** i * (1 - p1) ** (n - i) for i in range(seuil, n + 1))
        if pw >= puissance:
            return n
    return None


print("\n  ET ON PEUT CHIFFRER LE VOLUME QU'IL FAUDRAIT, une fois cette condition remplie. Pour")
print("  separer au seuil de 5 % avec 80 % de puissance la loi sans memoire de la loi")
print("  auto-evitante, sur les ecarts que le modele predit lui-meme a la section 4 :")
print(f"  {'triplet':<18}{'sans mem.':>11}{'auto-evit.':>12}{'n requis':>10}")
besoins = []
for lib, p0, p1, _r in gonf:
    n = n_min(p0, p1)
    besoins.append(n)
    print(f"  {lib:<18}{p0:>11.3f}{p1:>12.3f}{str(n):>10}")
bmin = min(b for b in besoins if b)
print(f"\n  Il faudrait donc de {bmin} a {max(b for b in besoins if b)} chemins passant par un MEME")
print(f"  pilier selon la sequence visee. Le corpus en fournit {n_tri} au total, tous piliers")
print("  confondus, et zero sur un pilier qui remplirait les deux conditions. L'ecart est d'un")
print("  a deux ordres de grandeur, et il porte sur la structure du corpus avant son volume.")

# =====================================================================================
titre("VERDICT")
# =====================================================================================
print("  1. LES SEQUENCES ETAIENT DEJA LA, implicites dans les ensembles d'aretes : "
      f"{n_tri} chemins")
print(f"     de longueur 2 pour {len(triplets)} sequences distinctes, plus une chaine de trois")
print("     aretes chez Equifax. Personne ne les avait composees.")
print("  2. LA QUESTION EST SANS OBJET SUR CE CORPUS, et il faut le dire ainsi : P2 n'emet")
print(f"     jamais ({emet[2]}/{len(ARETES)}) et P1 n'est jamais cible ({recoit[1]}/{len(ARETES)}),")
print("     donc les deux sequences comparees ont un support vide. Ensemble vide, pas")
print("     intervalle large.")
print("  3. DEUX RELATIONS A NE PAS CONFONDRE : le corpus code le conditionnement DIRECT, non")
print("     la precedence. La cloture transitive ajoute deux paires, dont (P1,P2), impliquees")
print("     par plusieurs incidents et codees par aucun. p_12 = 0 est correct pour W, qui est un")
print("     transfert direct, et ne dit rien de l'ordre d'arrivee.")
print(f"  4. L'ACYCLICITE N'EST PAS UN RESULTAT : {100*acy/tot:.0f} % des orientations d'un tel graphe")
print("     sont acycliques, donc on l'obtiendrait deux fois sur trois au hasard. Le nul par")
print("     observation la rendrait ecrasante, mais il teste la constance du codeur et non")
print("     l'existence d'une direction : c'est le mauvais nul.")
print(f"  5. LE MODELE PREDIT LES TRIPLETS, il n'est pas sans memoire : l'auto-evitement gonfle")
print(f"     la loi du successeur de {g_moy:.2f}x en moyenne, et c'est une borne basse. Les triplets")
print("     ne sont donc PAS redondants avec les paires, contrairement a ce qu'on suppose")
print("     spontanement, et la loi exacte des sequences est publiee ici pour la premiere fois.")
print("  6. MAIS LE TEST N'A AUCUNE PUISSANCE, et la cause est structurelle : le test exige un")
print("     pilier a la fois ATTEINT et a DEUX successeurs, et les deux manques sont exclusifs")
print("     dans ce corpus. P1 a trois successeurs mais n'est jamais atteint ; P3, P4 et P5 sont")
print("     atteints mais n'ont qu'un successeur. Un corpus plus grand du meme genre n'y")
print(f"     changerait rien ; il faudrait de {bmin} a {max(b for b in besoins if b)} chemins par un pilier qualifie.")
print("  7. DONC LES SEQUENCES NE RESSERRENT PAS LA BANDE D'IDENTIFICATION, et l'ignorance sur")
print("     la direction reste ce qu'elle est, graduee par le script 30. C'est un diagnostic")
print("     avec un critere de reouverture : un pilier a deux successeurs, repete.")

# =====================================================================================
# figure S31
# =====================================================================================
mpl.rcParams.update({
    "font.family": ["DejaVu Sans", "Segoe UI", "sans-serif"], "font.size": 11,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "axes.edgecolor": "#dcdcdc",
    "axes.linewidth": 0.8, "text.color": "#1b1e30", "axes.labelcolor": "#223e55",
    "xtick.color": "#595959", "ytick.color": "#595959", "axes.grid": False,
})
INK, INK2, MUTED = "#1b1e30", "#223e55", "#595959"
ACCENT, BLUE, GREEN = "#a6002e", "#2b559f", "#009a94"

# LARGEUR DE TRACE RAMENEE A LA LARGEUR D'IMPRESSION. Le memoire imprime
# cette figure sur 7,27 pouces : tracee plus large, elle subissait une
# reduction qui faisait tomber ses etiquettes sous 6 points. Le rapport
# largeur sur hauteur est conserve, donc la figure occupe la meme place
# sur la page ; c'est son texte qui y prend plus de place.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.9, 3.34))

# (a) le graphe agrege, en niveaux topologiques, avec les multiplicites
pos = {}
for lv, niveau in enumerate(ordre):
    for i, j in enumerate(niveau):
        pos[j] = (lv, (len(niveau) - 1) / 2.0 - i)
for (a, b) in obs:
    m = mult[tuple(sorted((a, b)))]
    x1, y1 = pos[a]
    x2, y2 = pos[b]
    ax1.annotate("", xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.7 + 0.55 * m,
                                 alpha=0.75, shrinkA=20, shrinkB=20))
    ax1.text(0.5 * (x1 + x2), 0.5 * (y1 + y2) + 0.09, str(m), fontsize=9, color=BLUE,
             ha="center", fontweight="bold")
for j, (x, y) in pos.items():
    src = emet[j] and not recoit[j]
    snk = recoit[j] and not emet[j]
    col = ACCENT if src else (GREEN if snk else "#ffffff")
    ax1.scatter([x], [y], s=1500, color=col, edgecolor=INK2, zorder=3, linewidths=0.9)
    ax1.text(x, y, f"P{j}", ha="center", va="center", fontsize=12, zorder=4,
             fontweight="bold", color="#ffffff" if (src or snk) else INK)
ax1.text(0.02, 0.97, "orange : source pure   ·   vert : puits pur\népaisseur et chiffre : "
         "nombre d'incidents", transform=ax1.transAxes, fontsize=8.5, color=INK2, va="top")
ax1.set_xlim(-0.6, len(ordre) - 0.4)
ax1.set_ylim(-1.3, 1.5)
ax1.axis("off")
ax1.set_title("(a)  Le corpus donne un ordre presque total,\net P2 n'émet jamais",
              fontsize=10.5, color=INK, pad=8)

# (b) la loi de succession observee, degeneree : c'est elle qui tue le test
emetteurs = [j for j in PIL if succ[j]]
xs = np.arange(len(emetteurs))
for i, j in enumerate(emetteurs):
    bas = 0.0
    tot_j = sum(succ[j].values())
    for kk, v in sorted(succ[j].items()):
        h = 100.0 * v / tot_j
        ax2.bar(i, h, bottom=bas, width=0.55, color=BLUE if len(succ[j]) == 1 else ACCENT,
                alpha=0.85, edgecolor="#fcfcfb", linewidth=1.2)
        ax2.text(i, bas + h / 2, f"P{kk}  ({v})", ha="center", va="center", fontsize=9,
                 color="#ffffff", fontweight="bold")
        bas += h
    # LES DEUX CONDITIONS DU TEST, cote a cote : c'est leur exclusivite qui est le resultat.
    ax2.text(i, 111, f"{len(succ[j])} cible" + ("s" if len(succ[j]) > 1 else ""),
             ha="center", fontsize=9, color=ACCENT if len(succ[j]) > 1 else BLUE,
             fontweight="bold")
    ax2.text(i, 104, f"atteint {recoit[j]} fois", ha="center", fontsize=8.5,
             color=BLUE if recoit[j] else ACCENT)
ax2.set_xticks(xs)
ax2.set_xticklabels([f"depuis P{j}" for j in emetteurs])
ax2.set_ylim(0, 126)
ax2.set_ylabel("part des transitions observées (%)", color=INK2)
# posee SOUS l'axe : a l'interieur du cadre elle traversait les barres.
ax2.text(0.5, -0.16, "le test exige les deux : être atteint ET avoir deux cibles. Aucun pilier "
         "ne remplit les deux.", transform=ax2.transAxes, ha="center", fontsize=9.5, color=INK,
         fontweight="bold")
ax2.set_title("(b)  Les deux conditions du test ne se rencontrent\njamais sur le même pilier",
              fontsize=10.5, color=INK, pad=8)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.91])
outdir = os.path.join(HERE, "figures")
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, "S31_sequences_ordonnees.png")
fig.savefig(path, dpi=200)
print("\nfigure ecrite :", path)
