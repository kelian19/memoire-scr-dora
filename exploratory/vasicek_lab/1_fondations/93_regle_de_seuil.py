# -*- coding: utf-8 -*-
"""93 - LE SEUIL DE LA CALIBRATION EST-IL UNE REGLE OU UN CHOIX ?

LA QUESTION EST CELLE QU'UN JURY POSE EN PREMIER SUR UN MODELE DE DEPASSEMENTS DE SEUIL.
La chaine publiee fixe u = 20,03 M EUR, soit le percentile 84,4 de la donnee courante, et le
memoire le presente comme la calibration figee. Un lecteur demande alors pourquoi CE seuil, et
la reponse ne peut pas etre << parce que c'est celui-la >>. Ce script confronte le seuil publie
a des regles de selection ecrites, et mesure la distance.

LE GEL INTERDIT DE CHANGER u, ET CE N'EST PAS UNE GENE. Le livrable n'est pas un nouveau seuil :
c'est de savoir si le seuil publie est celui qu'une regle aurait choisi, et sinon de chiffrer
l'ecart. C'est exactement le traitement du taux de depassement gele, un ecart imprime plutot
qu'une correction silencieuse.

TROIS REGLES SONT MISES EN OEUVRE, ET UNE QUATRIEME EST REFUSEE AVEC SON MOTIF.
  (A) la plus BASSE valeur non rejetee par un test d'adequation, dans l'esprit du test de
      score sequentiel : on descend tant que la GPD reste acceptable, puisque descendre
      augmente le nombre d'exces donc la precision ;
  (B) la fenetre de STABILITE : la plus basse valeur au-dessus de laquelle tous les indices de
      queue estimes restent dans l'intervalle de confiance de celui-la ;
  (C) le compromis biais-variance lu directement, en imprimant le quantile publie et son
      ecart-type bootstrap le long du balayage. Ce n'est pas une regle mais l'arbitrage
      lui-meme, et il vaut mieux qu'une regle si les deux premieres se contredisent.
  (D) REFUSEE : les regles de minimisation de l'erreur quadratique asymptotique par double
      bootstrap, dont celle de Danielsson et de Haan. Motif imprime en fin de script, et ce
      n'est pas le temps de calcul.

CE QU'IL TROUVE, EN TROIS LIGNES, ET LE RESULTAT EST FAVORABLE. La regle de stabilite selectionne
EXACTEMENT le seuil publie, a 0,8 % pres et a quantile identique, et ce seuil est aussi le mieux
ajuste du balayage, sa p-value d'Anderson-Darling valant 0,971 contre 0,043 a 0,954 ailleurs. La
regle la plus permissive descendrait plus bas et donnerait un quantile SUPERIEUR de 16 %, donc le
seuil publie n'est pas celui qui maximise le chiffre du memoire. Et le compromis biais-variance
attendu n'existe pas sur cette grandeur : descendre le seuil degrade a la fois le biais et la
precision, l'ecart-type du quantile etant divise par 3,5 en remontant.

C'EST UN DIAGNOSTIC. config.py n'est ni lu pour etre ecrit ni touche.
"""

import os
import sys

import numpy as np
from scipy.stats import genpareto

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for _p in (REPO, LAB):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import derive_severite as ds  # noqa: E402
from src.utils.config import OPRISK  # noqa: E402

PERCENTILES = (0.70, 0.75, 0.80, 0.825, 0.844, 0.85, 0.875, 0.90, 0.925, 0.95)
B_GOF = 800          # bootstrap parametrique des p-values d'adequation, comme le script 47
B_SE = 400           # bootstrap non parametrique de l'ecart-type du quantile
SEED = 20260908
NIV = 0.995


def titre(t):
    print()
    print("=" * 88)
    print(t)
    print("=" * 88)


def ad_stat(x, c, s):
    """Statistique d'Anderson-Darling contre une GPD de parametres (c, s). Script 47."""
    z = np.clip(genpareto.cdf(np.sort(x), c, scale=s), 1e-12, 1 - 1e-12)
    m = z.size
    i = np.arange(1, m + 1)
    return -m - np.sum((2 * i - 1) / m * (np.log(z) + np.log(1 - z[::-1])))


def q_pot(q, xi, sig, u, p_u):
    return u + (sig / xi) * (((1.0 - q) / p_u) ** (-xi) - 1.0)


d = ds.charger_pertes()
x = np.sort(d["loss_eur"].values)
n = x.size
u_pub = float(OPRISK["seuil_u_eur"])

titre("0. LE SEUIL PUBLIE, SITUE DANS SON ECHANTILLON")
print(f"Echantillon : {n} incidents du perimetre cyber x finance, convention de severite du "
      "projet.")
print(f"Seuil publie : {u_pub} M EUR, soit le percentile "
      f"{100 * float((x <= u_pub).mean()):.1f}, laissant "
      f"{int((x > u_pub).sum())} exces.")
print(f"Grandeur cible du balayage : le quantile de severite a {100*NIV:.1f} %, que le memoire")
print(f"publie a {OPRISK['var_995']} M EUR.")


# ---------------------------------------------------------------------------
# 1. Le balayage, avec adequation et precision
# ---------------------------------------------------------------------------

titre("1. BALAYAGE DES SEUILS CANDIDATS : ADEQUATION, ESTIMATION, PRECISION")
print("Pour chaque percentile candidat : ajustement GPD libre sur les exces, taux de depassement")
print("COMPTE, p-value d'Anderson-Darling par bootstrap parametrique, puis quantile a 99,5 % et")
print("son ecart-type par re-echantillonnage non parametrique des exces.")
print()
print("perc.    seuil  n exc.      xi    sigma   p AD    q 99,5 %   ecart-type")

rng = np.random.default_rng(SEED)
lignes = []
for p in PERCENTILES:
    u = float(np.quantile(x, p))
    exc = x[x > u] - u
    if exc.size < 20:
        continue
    xi, _, sig = genpareto.fit(exc, floc=0.0)
    p_u = float((x > u).mean())
    q = q_pot(NIV, xi, sig, u, p_u)

    # adequation : loi nulle par bootstrap parametrique, parametres reajustes a chaque tirage
    obs = ad_stat(exc, xi, sig)
    nul = []
    for _ in range(B_GOF):
        xb = genpareto.rvs(xi, scale=sig, size=exc.size, random_state=rng)
        try:
            cb, _, sb = genpareto.fit(xb, floc=0.0)
            nul.append(ad_stat(xb, cb, sb))
        except Exception:
            pass
    p_ad = float((np.array(nul) >= obs).mean())

    # precision du quantile : re-echantillonnage des exces, seuil et taux tenus fixes
    qs = []
    for _ in range(B_SE):
        eb = rng.choice(exc, size=exc.size, replace=True)
        try:
            cb, _, sb = genpareto.fit(eb, floc=0.0)
            if cb > 0:
                qs.append(q_pot(NIV, cb, sb, u, p_u))
        except Exception:
            pass
    se = float(np.std(qs, ddof=1)) if len(qs) > 10 else float("nan")

    lignes.append(dict(p=p, u=u, n=exc.size, xi=xi, sig=sig, p_ad=p_ad, q=q, se=se))
    marque = "  <- seuil publie" if abs(u - u_pub) < 0.6 else ""
    print(f"  {100*p:5.1f}  {u:7.2f}   {exc.size:4d}  {xi:6.4f}  {sig:7.2f}  {p_ad:6.3f}  "
          f"{q:9.1f}   {se:9.1f}{marque}")

print()
print("REGLE (C), ET LE COMPROMIS ATTENDU N'EXISTE PAS SUR CES DONNEES. On ecrit par habitude que")
print("descendre le seuil gagne des exces donc de la precision, et que le remonter reduit le")
print("biais mais paye en variance. La colonne d'ecart-type dit l'inverse :")
se_bas, se_haut = lignes[0]["se"], lignes[-1]["se"]
print(f"  ecart-type du quantile au percentile {100*lignes[0]['p']:.1f} : {se_bas:.1f} M EUR "
      f"sur {lignes[0]['n']} exces")
print(f"  ecart-type du quantile au percentile {100*lignes[-1]['p']:.1f} : {se_haut:.1f} M EUR "
      f"sur {lignes[-1]['n']} exces")
print(f"  soit une DIVISION par {se_bas / se_haut:.1f} en remontant le seuil, alors que le nombre")
print("  d'exces est divise par presque six.")
print()
print("LE MOTIF EST INTELLIGIBLE ET IL EST PROPRE A LA GRANDEUR VISEE. Le quantile a 99,5 % est")
print("exponentiellement sensible a l'indice de queue, et descendre le seuil GONFLE cet indice :")
print(f"  il passe de {lignes[-1]['xi']:.4f} au percentile {100*lignes[-1]['p']:.1f} a "
      f"{lignes[0]['xi']:.4f} au percentile {100*lignes[0]['p']:.1f}, donc au-dessus de UN,")
print("  ou l'esperance de la severite n'existe plus. Les exces gagnes en descendant sont des")
print("  pertes qui n'appartiennent pas encore a la queue : ils ajoutent du BIAIS, et le biais")
print("  se propage dans l'ecart-type par le levier de l'indice.")
print()
print("CONSEQUENCE POUR LE CHOIX. Il n'y a donc rien a arbitrer entre biais et precision sur cette")
print("grandeur : les deux vont dans le meme sens et le seuil doit se choisir sur l'ADEQUATION et")
print("la STABILITE, non sur la precision. C'est ce que font les regles (A) et (B).")


# ---------------------------------------------------------------------------
# 2. Regle (A) : la plus basse valeur non rejetee
# ---------------------------------------------------------------------------

titre("2. REGLE (A) : LA PLUS BASSE VALEUR NON REJETEE PAR L'ADEQUATION")
print("La GPD n'est justifiee qu'au-dessus d'un seuil assez haut. On descend donc tant que le")
print("test ne rejette pas, puisque descendre gagne des exces. La regle retient le plus BAS")
print("seuil dont la p-value depasse 5 %, ET dont tous les seuils superieurs passent aussi :")
print("un unique passage isole serait un accident d'echantillon, non une frontiere.")
print()
ok = [L for L in lignes if L["p_ad"] > 0.05]
u_A = None
for i, L in enumerate(lignes):
    if L["p_ad"] > 0.05 and all(M["p_ad"] > 0.05 for M in lignes[i:]):
        u_A = L
        break
if u_A is None:
    print("  aucun seuil du balayage ne satisfait la regle")
else:
    print(f"  seuil retenu par (A) : {u_A['u']:.2f} M EUR, percentile {100*u_A['p']:.1f}, "
          f"{u_A['n']} exces")
    print(f"  quantile a 99,5 %    : {u_A['q']:.1f} M EUR, ecart-type {u_A['se']:.1f}")
    print(f"  p-value d'Anderson-Darling : {u_A['p_ad']:.3f}")
print(f"  nombre de seuils du balayage non rejetes : {len(ok)} sur {len(lignes)}")


# ---------------------------------------------------------------------------
# 3. Regle (B) : la fenetre de stabilite
# ---------------------------------------------------------------------------

titre("3. REGLE (B) : LA FENETRE DE STABILITE DE L'INDICE DE QUEUE")
print("Un seuil est admissible si l'indice de queue cesse de deriver au-dessus de lui. On")
print("retient donc le plus BAS seuil tel que tous les indices estimes au-dessus tombent dans")
print("l'intervalle de confiance a 90 % de celui-la, obtenu par re-echantillonnage des exces.")
print()
print("perc.  xi     IC90 de xi        xi superieurs dedans ?")
u_B = None
for i, L in enumerate(lignes):
    exc = x[x > L["u"]] - L["u"]
    xb = []
    for _ in range(B_SE):
        eb = rng.choice(exc, size=exc.size, replace=True)
        try:
            cb, _, _s = genpareto.fit(eb, floc=0.0)
            xb.append(cb)
        except Exception:
            pass
    lo, hi = np.quantile(xb, [0.05, 0.95])
    sup = [M["xi"] for M in lignes[i + 1:]]
    dedans = all(lo <= v <= hi for v in sup) if sup else True
    if dedans and u_B is None:
        u_B = dict(L, lo=float(lo), hi=float(hi))
    print(f"  {100*L['p']:5.1f}  {L['xi']:.4f}  [{lo:.4f} ; {hi:.4f}]   "
          f"{'oui' if dedans else 'non'}"
          + ("   <- retenu par (B)" if u_B is not None and u_B["p"] == L["p"] else ""))
if u_B is None:
    print("  aucun seuil du balayage ne satisfait la regle")


# ---------------------------------------------------------------------------
# 4. Le seuil publie contre les deux regles
# ---------------------------------------------------------------------------

titre("4. LE SEUIL PUBLIE CONTRE LES DEUX REGLES")
L_pub = min(lignes, key=lambda L: abs(L["u"] - u_pub))
print(f"seuil publie : {u_pub} M EUR (ligne du balayage la plus proche : percentile "
      f"{100*L_pub['p']:.1f}, {L_pub['u']:.2f})")
for etiq, R in (("regle (A), plus bas non rejete", u_A), ("regle (B), fenetre de stabilite", u_B)):
    if R is None:
        print(f"  {etiq:<34} : sans solution sur le balayage")
        continue
    print(f"  {etiq:<34} : {R['u']:7.2f} M EUR, percentile {100*R['p']:.1f}")
    print(f"    ecart au seuil publie              : {R['u'] - u_pub:+7.2f} M EUR, soit "
          f"{100*(R['u']/u_pub - 1):+.1f} %")
    print(f"    quantile a 99,5 % qui en resulte   : {R['q']:7.1f} contre "
          f"{L_pub['q']:.1f} au seuil publie, soit {100*(R['q']/L_pub['q'] - 1):+.1f} %")
    print(f"    et contre le chiffre PUBLIE ({OPRISK['var_995']}) : "
          f"{100*(R['q']/OPRISK['var_995'] - 1):+.1f} %")
print()
print("LES DEUX DERNIERES LIGNES NE MESURENT PAS LA MEME CHOSE, et il ne faut pas les confondre.")
print("La comparaison au quantile du seuil publie isole l'effet du SEUIL, les deux etant calcules")
print("par la meme regle avec un taux de depassement COMPTE. La comparaison au chiffre publie y")
print("ajoute le defaut du taux de depassement gele, deja declare au chapitre 13. Melanger les")
print("deux ferait porter au seuil un ecart qui ne lui appartient pas.")


# ---------------------------------------------------------------------------
# 5. La regle refusee, et le motif
# ---------------------------------------------------------------------------

titre("5. LA REGLE REFUSEE : LE DOUBLE BOOTSTRAP D'ERREUR QUADRATIQUE ASYMPTOTIQUE")
print("Les regles de reference pour choisir le nombre d'exces minimisent une erreur quadratique")
print("ASYMPTOTIQUE par double bootstrap, sur deux sous-echantillons emboites. Elles ne sont pas")
print("employees ici, et le motif n'est ni le temps de calcul ni la difficulte de mise en oeuvre.")
print()
n_exc_pub = int((x > u_pub).sum())
print(f"  exces disponibles au seuil publie : {n_exc_pub}")
print(f"  taille de l'echantillon complet   : {n}")
print("  ces regles reposent sur un regime ou le nombre d'exces tend vers l'infini plus")
print("  lentement que la taille de l'echantillon, et leurs sous-echantillons emboites sont")
print("  d'ordre n puissance 3/4 puis son carre divise par n, soit ici")
print(f"  {int(round(n ** 0.75))} puis {int(round((n ** 0.75) ** 2 / n))} observations.")
print("  A cette taille l'estimateur du minimum est lui-meme tres bruite, et la regle rendrait")
print("  un nombre AYANT L'APPARENCE D'UNE REGLE SANS EN AVOIR LA GARANTIE.")
print()
print("C'EST LE MEME ARBITRAGE QUE DEUX AUTRES DEJA RENDUS DANS CE PROJET, et la coherence")
print("compte : l'elicitation a ete ecartee parce qu'un panel faiblement calibre introduirait une")
print("incertitude NON DECLARABLE en echange d'une ignorance mesuree, et l'ancrage de la")
print("propagation sur les publications prudentielles a ete refuse parce que le resultat aurait")
print("l'apparence d'un calibrage sans en etre un. Une regle dont les conditions d'application ne")
print("sont pas reunies est moins bonne qu'un choix declare et confronte.")


titre("VERDICT")
print("Ecrit APRES lecture des sorties. Le commentaire de la section 1 annoncait un compromis")
print("biais-variance : la colonne d'ecart-type l'a dementi, et c'est elle qui tranche.")
print()
p_max = max(lignes, key=lambda L: L["p_ad"])
print("1. LA REGLE DE STABILITE SELECTIONNE EXACTEMENT LE SEUIL PUBLIE, et c'est le resultat")
print(f"   principal. Elle retient {u_B['u']:.2f} M EUR quand le memoire publie {u_pub}, soit")
print(f"   {100*(u_B['u']/u_pub - 1):+.1f} %, et le quantile qui en resulte est identique au")
print("   centieme de pour cent. Le seuil publie est donc precisement le plus bas au-dessus")
print("   duquel l'indice de queue cesse de deriver. Ce n'est pas un critere fabrique apres coup :")
print("   c'est le balayage de seuil que le chapitre 06 publie deja, employe comme regle de")
print("   selection au lieu d'etre seulement decrit.")
print()
print("2. ET C'EST AUSSI LE SEUIL LE MIEUX AJUSTE DU BALAYAGE. Sa p-value d'Anderson-Darling vaut")
print(f"   {L_pub['p_ad']:.3f}, la plus haute des {len(lignes)} candidats "
      f"(le suivant est {sorted(L['p_ad'] for L in lignes)[-2]:.3f}).")
print("   L'adequation et la stabilite designent donc le meme point, ce qui n'etait pas garanti :")
print("   deux criteres independants pouvaient parfaitement se contredire.")
print()
print("3. LA REGLE LA PLUS PERMISSIVE DESCENDRAIT PLUS BAS ET DONNERAIT DAVANTAGE DE CAPITAL, ce")
print("   qui est l'argument le plus fort contre un soupcon de choix opportuniste. La regle (A)")
print(f"   retient {u_A['u']:.2f} M EUR, soit {abs(100*(u_A['u']/u_pub - 1)):.1f} % SOUS le seuil "
      f"publie, et le quantile y vaut {u_A['q']:.1f}")
print(f"   contre {L_pub['q']:.1f}, soit {100*(u_A['q']/L_pub['q'] - 1):+.1f} %. Le seuil publie "
      "n'est donc PAS celui qui")
print("   maximise le chiffre du memoire : parmi les candidats acceptables, il est du cote qui le")
print("   minore. Une assurance verbale ne vaudrait pas cette comparaison.")
print()
print("4. LE COMPROMIS BIAIS-VARIANCE N'EXISTE PAS SUR CETTE GRANDEUR, contre l'attente. L'ecart-")
print(f"   type du quantile est divise par {lignes[0]['se'] / lignes[-1]['se']:.1f} en remontant "
      "le seuil, alors que le nombre d'exces")
print("   est divise par presque six : descendre degrade a la fois le biais et la precision,")
print("   parce que l'indice de queue gonfle et que le quantile en depend exponentiellement. Il")
print("   n'y a donc rien a arbitrer, et le seuil se choisit sur l'adequation et la stabilite.")
print()
print("5. LES REGLES D'ERREUR QUADRATIQUE ASYMPTOTIQUE SONT REFUSEES AVEC LEUR MOTIF, section 5,")
print("   et le motif est de coherence : une regle dont les conditions d'application ne sont pas")
print("   reunies rendrait un nombre ayant l'apparence d'une regle sans en avoir la garantie.")
print("   Meme arbitrage que celui qui a ecarte l'elicitation et l'ancrage de la propagation.")
print()
print("CE QUE CE SCRIPT NE DIT PAS, ET IL Y A TROIS RESERVES.")
print("  - le balayage porte sur UN echantillon. Les deux regles s'accordent ici, mais rien ne")
print("    garantit qu'elles s'accorderaient sur un autre tirage, et leur concordance est un")
print("    constat, non une propriete ;")
print(f"  - le faible ecart-type du percentile {100*lignes[-1]['p']:.1f} ne doit PAS se lire "
      f"comme de la precision : il")
print(f"    est estime sur {lignes[-1]['n']} exces, donc lui-meme mal connu. Un ecart-type est "
      "connu a environ")
print("    la racine de deux fois le nombre d'observations pres, ce qui interdit de comparer les")
print("    deux extremites du balayage sur cette seule colonne ;")
print("  - et rien de ceci ne change le seuil publie, que le gel interdit de toucher. Le livrable")
print("    est la CONFRONTATION, pas un nouveau seuil.")

print()
print("EXIT 0")
