#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Non-régression des chiffres du mémoire.

MOTIVATION. Le script 01 a longtemps imprimé des conclusions codées en dur, contredites
par ses propres chiffres affichés juste au-dessus. Le contrôle qualité reposait sur la
relecture, qui ne passe pas à l'échelle sur plusieurs centaines de nombres. Ce script
remplace la relecture par une vérification : chaque nombre publié dans un chapitre est
recherché dans la sortie du script qui est censé le produire.

PRINCIPE. Pour chaque section du chapitre, on relève les scripts cités (\\texttt{16},
\\texttt{20}, ...) et tous les nombres du texte. Un nombre est CONFIRMÉ s'il apparaît,
à la tolérance d'arrondi près, dans la sortie d'au moins un des scripts cités par sa
section. Sinon il est signalé.

CE QUE CE SCRIPT NE FAIT PAS. Il ne vérifie pas qu'un nombre est au bon endroit ni qu'il
veut dire ce que la phrase prétend : un nombre confirmé peut être mal commenté. Il élimine
la classe d'erreurs la plus fréquente et la plus embarrassante (le chiffre périmé ou
inventé), pas toutes.

Usage :
    python verif_chiffres.py <dossier_des_sorties>
où le dossier contient 16.txt, 20.txt, ... produits en relançant les scripts cités.
"""

import os
import re
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
CHAPITRE_DEFAUT = os.path.join(ICI, "chapitres", "12_resultats.tex")

# Nombres qui ne sont pas des résultats : niveaux de confiance, entiers d'énumération,
# millésimes. Les comparer n'aurait aucun sens.
#
# EXEMPTES, ET DESORMAIS COMPTES COMME TELS. La version precedente les retirait en silence :
# le denominateur du taux de confirmation n'etait donc pas « les nombres publies » mais
# « les nombres publies moins une liste », et le rapport ne disait pas laquelle. Un taux de
# 99,6 % sur une population filtree sans le dire n'est pas un controle, c'est une mesure de
# soi-meme. Le rapport imprime maintenant les trois lignes : verifies, exemptes par motif,
# non confirmes.
#
# L'EXEMPTION SE JUGE SUR LE CONTEXTE, PAS SUR LA VALEUR, et c'est la que la liste pechait.
# Deux entrees y figuraient qui designent aussi des resultats, et n'ont donc jamais ete
# verifiees nulle part dans le memoire :
#   - 2,5, retire ici : c'est le facteur d'incertitude de calibration sur la VaR, cite aux
#     chapitres 05, 06, 12 et 13, et c'est precisement le nombre sur lequel portait
#     l'arbitrage 2,5 contre 2,6. Le harnais ne pouvait pas le trancher : il ne le regardait
#     pas.
#   - 19, retire ici : tantot l'article 19 de la directive, tantot la part de 19 % du pilier
#     P2 sous PRC, tantot les dix-neuf prestataires TIC critiques designes par les AES.
# Les deux premieres acceptions se confirment sur les sorties ; la troisieme, l'article, est
# dans un \texttt{} ou dans une phrase sans script cite, donc hors champ de toute facon.
NIVEAUX = {99.5, 0.995, 97.5, 90.0, 95.0}     # niveaux de confiance et de quantile
ENUM = {1.0, 2.0, 3.0, 4.0, 5.0}              # indices de pilier, compteurs d'items
# Millesimes. La borne basse descend a 1900 pour couvrir les dates de theoremes citees dans
# le chapitre socle (Fisher-Tippett-Gnedenko 1928, 1943 ; Balkema-de Haan-Pickands 1974).
# Un montant de cet ordre s'ecrit toujours avec un separateur de milliers dans le memoire,
# donc la confusion avec un resultat reste improbable ; le cas echeant il faut le verifier
# a la main, ce que la liste de travail signale.
ANNEES = set(range(1900, 2036))


def extrait(txt, latex=True):
    """Tous les nombres d'un fragment, normalisés (8\\,301 -> 8301 ; 0{,}92 -> 0.92).

    Avant exemption : c'est `nombres` qui filtre, et `exemption` qui dit pourquoi.

    On retire d'abord la MISE EN PAGE : un \\arraystretch de 1,25 ou une largeur de
    figure de 0,9\\linewidth ne sont pas des résultats et les compter comme tels
    fabriquerait de fausses alertes.
    """
    t = txt
    # LES COMMENTAIRES LATEX NE SONT PAS DU TEXTE PUBLIE. Sans ce filtre, un commentaire
    # de migration comme "% MIGRE depuis main.tex (plages 787-932)" fabrique quatre
    # nombres a confirmer qui n'apparaissent nulle part dans le PDF. C'etait la premiere
    # cause de faux positifs du chapitre socle.
    #
    # A NE SURTOUT PAS APPLIQUER AUX SORTIES DE SCRIPTS : elles ecrivent les pourcentages
    # avec un %, et couper la ligne au premier % y detruisait la moitie du pool de
    # reference. D'ou le drapeau `latex`, qui vaut False pour les fichiers NN.txt.
    if latex:
        t = re.sub(r"(?<!\\)%.*", " ", t)
    t = re.sub(r"\\renewcommand\{[^}]*\}\{[^}]*\}", " ", t)
    # LES ESPACEMENTS SONT DE LA MISE EN PAGE, au meme titre que \arraystretch : un
    # \vspace{0.25cm} versait un 0,25 a confirmer dans chaque section qui aere un tableau.
    t = re.sub(r"\\[vh]space\*?\{[^}]*\}", " ", t)
    t = re.sub(r"\\includegraphics\[[^\]]*\]", " ", t)
    t = re.sub(r"\\(?:label|ref|eqref|cite[tp]?)\{[^}]*\}", " ", t)
    t = re.sub(r"\[[^\]]*(?:width|height|scale)[^\]]*\]", " ", t)
    # Un NUMERO DE SCRIPT n'est pas un resultat. Il vit toujours dans un \texttt{}, comme
    # les noms de fichier et de base : aucun chiffre publie n'est en fonte a chasse fixe.
    # Sans ce filtre, citer le script 27 fabrique un "27" a confirmer, et le rapport de
    # verification se remplit de faux positifs qui masquent les vraies erreurs.
    t = re.sub(r"\\texttt\{[^}]*\}", " ", t)
    # Les specifications de filet de tableau (\cmidrule{6-8}, \cline{2-4}) sont de la mise
    # en page : elles produisaient un 6 et un -8 dans chaque section a tableau.
    t = re.sub(r"\\c(?:midrule|line)\s*(?:\([^)]*\))?\s*\{[^}]*\}", " ", t)
    t = re.sub(r"\\multicolumn\{\d+\}", " ", t)
    # LA FRACTION COURTE. \tfrac12 s'ecrit sans accolades ; le nettoyage generique des
    # commandes LaTeX, quelques lignes plus bas, en retirait le \tfrac et laissait « 12 ».
    # Un demi devenait donc le nombre douze, a confirmer dans les sorties de scripts. Le
    # chapitre identifiabilite en fabriquait cinq a lui seul, sur la decomposition
    # P = S_pi + A_pi. La forme longue \frac{1}{2} n'a pas le probleme : elle laisse 1 et 2,
    # deja neutralises par IGNORE.
    t = re.sub(r"\\[tdc]?frac\s*\d\d", " ", t)
    # LES LARGEURS DE COLONNE. Un \begin{tabular}{@{}p{4.3cm} r r r r@{}} produisait 4.3,
    # de la mise en page au meme titre qu'un \arraystretch. Meme motif pour m{} et b{}.
    t = re.sub(r"\b[pmb]\{\s*[\d.]+\s*(?:cm|mm|in|pt|em|ex|\\[a-zA-Z]+)\s*\}", " ", t)
    # LES LARGEURS DE COLONNE EN FRACTION DE LIGNE, et la docstring les annoncait deja a tort.
    # Un \includegraphics[width=0.96\linewidth] etait bien neutralise, son argument optionnel
    # etant retire en bloc, mais PAS un \begin{column}{0.46\linewidth} : le 0,46 partait dans le
    # pool a confirmer. Il ne s'y voyait pas tant que la valeur tombait par hasard sur une sortie
    # de script (0,45 et 0,52 se confirmaient ainsi dans le deck du 14), et il ressortait en
    # fausse alerte des qu'on choisissait une autre largeur. Meme classe que \tfrac12 et p{4.3cm}.
    t = re.sub(r"[\d.]+\s*\\(?:line|text|column|paper)(?:width|height)\b", " ", t)
    # LES CODES DE COULEUR HEXADECIMAUX. Un \definecolor{navy}{HTML}{1F3864} versait 3864 dans
    # le pool a confirmer, la partie alphabetique etant simplement sautee par le motif
    # d'extraction. Trouve le 13 aout sur les notes de suivi, qui definissent six couleurs
    # chacune. Meme classe que p{4.3cm} : de la mise en page lue comme un resultat.
    t = re.sub(r"\\definecolor\{[^}]*\}\{[^}]*\}\{[^}]*\}", " ", t)
    # LES PENALITES DE CESURE. Un \hyphenpenalty=10000 dans une specification de colonne versait
    # 10000 dans le pool, deux fois par table. Meme classe, meme remede.
    t = re.sub(r"\\(?:hyphen|exhyphen|widow|club|binop|rel)penalty\s*=?\s*-?\d+", " ", t)
    # LE MELANGE DE COULEURS DE xcolor. Un colframe=navy!80 versait 80 dans le pool. La syntaxe
    # « identifiant ! entier » n'apparait pas en prose, le motif est donc sans risque.
    t = re.sub(r"\b[A-Za-z]+!\d+\b", " ", t)
    # LES CLES DIMENSIONNEES des options de boite et de tableau : boxrule=0.9pt, arc=2pt,
    # left=8pt. Elles n'etaient pas couvertes par la regle sur p{4.3cm}, et un 0,9 leake dans le
    # pool s'y confirme par hasard sans qu'on le voie jamais, ce qui est le pire des cas.
    t = re.sub(r"\b[A-Za-z]+\s*=\s*-?[\d.]+\s*(?:pt|mm|cm|em|ex|in|bp|sp)\b", " ", t)
    # LA NOTATION SCIENTIFIQUE N'EST PAS DEUX NOMBRES. « p \approx 10^{-30} » ne publie ni un
    # dix ni un trente : il publie un ordre de grandeur, et le motif d'extraction en tirait un
    # « 10 » que rien ne pouvait confirmer. Meme classe que \tfrac12 et p{4.3cm} : un artefact
    # de lecture, pas un chiffre du memoire. On ne neutralise que les puissances NEGATIVES de
    # dix, qui sont toujours de la notation scientifique ; 2^{10} garde son exposant, qui lui
    # compte bien dix parametres libres.
    t = re.sub(r"\b10\s*\^\s*\{?\s*-\s*\d+\s*\}?", " ", t)
    # UN NUMERO D'ARTICLE N'EST PAS UN RESULTAT, et c'est une exemption de CONTEXTE, la seule
    # sorte qui vaille : on ne retire pas la valeur 18 partout, on retire « art. 18 ». La
    # difference compte, c'est en exemptant sur la valeur que le facteur 2,5 avait echappe au
    # controle pendant des semaines.
    t = re.sub(r"\b(?:art\.?|articles?)\s*~?\s*\d+(?:\s*(?:,|et|à|a|--?|–)\s*\d+)*", " ", t,
               flags=re.IGNORECASE)
    # UN NUMERO DE REGLEMENT N'EST PAS UN RESULTAT NON PLUS, meme exemption de CONTEXTE. Le
    # resume ecrit « le reglement DORA (UE 2022/2554) » et le harnais y lisait 2022 puis 2554 :
    # le millesime etait exempte par motif, le numero d'ordre non, et il ressortait comme un
    # nombre du memoire a confirmer. Aucune sortie de script ne l'imprime, et aucune ne le
    # devrait. Le motif ne retire que la forme ANNEE/NUMERO, jamais la valeur 2554 ailleurs :
    # c'est la meme discipline que pour les articles, on exempte l'occurrence et pas le nombre.
    t = re.sub(r"\b(?:19|20)\d{2}\s*/\s*\d{1,5}\b", " ", t)
    t = t.replace("\\,", "").replace("~", " ").replace("{,}", ".")
    t = re.sub(r"\\[a-zA-Z]+", " ", t)          # commandes LaTeX restantes
    if not latex:
        # L'ECART-TYPE N'EST PAS UN NOMBRE NEGATIF. Les sorties ecrivent « 9.5+-2.4 » pour
        # 9,5 plus ou moins 2,4 ; le motif d'extraction y voyait 9.5 puis MOINS 2.4, et le
        # « 2,4 » du memoire ressortait non confirme alors que le script l'imprime.
        t = t.replace("+-", "  ").replace("+/-", "  ")
        # UN NUMERO DE SCRIPT N'EST PAS UN RESULTAT, DES DEUX COTES. Le filtre \texttt{} plus
        # haut l'assure pour le memoire ; rien ne l'assurait pour les sorties, ou une ligne
        # « P1 en tete de priorite (script 30) » versait un 30 dans le pool de reference. Ce
        # 30 confirmait alors, dans toute section citant ce script, le « minimum de 30 exces »
        # et le « k = 30 » du graphe de Hill du chapitre socle, deux valeurs qui n'ont aucun
        # rapport avec lui et que le harnais avait raison de signaler.
        t = re.sub(r"\bscripts?\s*n?o?s?\.?\s*\d+[a-z]?(?:\s*(?:,|et|;)\s*\d+[a-z]?)*",
                   " ", t, flags=re.IGNORECASE)
        # LE SEPARATEUR DE MILLIERS DES SORTIES DE SCRIPT. Le motif d'extraction ci-dessous
        # ne franchit pas la virgule : « 8,122.9 » y devenait DEUX nombres, 8 et 122.9, et le
        # « 8123 » du memoire ressortait donc non confirme alors que le script l'imprimait.
        # Toute la classe des montants a quatre chiffres et plus etait touchee, ce qui noyait
        # les vraies erreurs sous les fausses alertes.
        #
        # AMBIGUITE ASSUMEE, ET C'EST LE POINT DELICAT. Une virgule entre chiffres peut aussi
        # etre un separateur DECIMAL francais, que certaines chaines de narration emploient
        # (« z = +5,12 »). On ne colle donc que les groupes de la forme anglo-saxonne stricte :
        # 1 a 3 chiffres commencant par un chiffre NON NUL, puis exactement 3 chiffres. Cela
        # exclut « 0,807 » et « 0,505 ». Il reste un cas indecidable, « 2,150 » voulant dire
        # 2,150 en decimal francais ; le risque est alors une confirmation a tort, pas un
        # chiffre faux dans le memoire, et il est prefereable a un detecteur ignore parce
        # qu'il crie trop souvent.
        t = re.sub(r"(?<![\d.,])([1-9]\d{0,2})((?:,\d{3})+)(?!\d)",
                   lambda m: m.group(1) + m.group(2).replace(",", ""), t)
        # LA VIRGULE DECIMALE FRANCAISE DES SORTIES, ET C'EST LE PLUS NUISIBLE DES DEUX SENS.
        # Les scripts narrent en francais : « +4 % a 99,9 % », « xi = 0,595 ». Une fois les
        # groupes de milliers anglo-saxons recolles juste au-dessus, toute virgule qui reste
        # entre deux chiffres est un separateur decimal. Sans cette ligne, « 99,9 » n'entrait
        # pas dans le pool sous la forme 99,9 : il y versait DEUX entiers, 99 et 9. Le pool
        # perdait donc la vraie valeur et gagnait deux fausses, dont un « 9 » qui confirmait
        # ensuite n'importe quel neuf du memoire. Corriger cela resserre le controle autant
        # qu'il l'elargit.
        t = re.sub(r"(?<=\d),(?=\d)", ".", t)
        # Notation scientifique des sorties. LA MANTISSE EST UN CHIFFRE PUBLIE, PAS L'EXPOSANT :
        # le memoire ecrit « p = 1,53 \cdot 10^{-5} », et c'est bien 1,53 qu'il faut confirmer.
        # On garde donc la mantisse et l'on jette l'exposant, qui versait sinon un -5 ou un -30
        # dans le pool. Supprimer le jeton entier, comme une premiere version le faisait, sortait
        # au contraire le 1,53 du chapitre identifiabilite : le filtre doit couper la notation,
        # pas la valeur.
        t = re.sub(r"\b(\d+(?:\.\d+)?)[eE][-+]?\d+\b", r" \1 ", t)
    t = t.replace("--", " ")
    out = []
    # UN MOINS ENTRE DEUX NOMBRES EST UNE SOUSTRACTION, PAS UN SIGNE. Dans « kappa = 1 -
    # 0,0103/0,0475 », le motif lisait « -0,0103 » et cherchait un nombre negatif que le
    # script imprime positif. On n'accepte donc le signe que s'il ne suit ni un chiffre ni
    # une parenthese fermante. Le tiret demi-cadratin des plages, « 2005--2022 », est
    # neutralise juste avant : sans cela il fabriquait l'annee negative -2022.
    for m in re.finditer(r"(?<![\d)])-?\d+(?:\.\d+)?", t):
        try:
            out.append(float(m.group(0)))
        except ValueError:
            continue
    return out


def exemption(v):
    """Motif pour lequel v n'est pas un résultat à confirmer, ou None s'il doit l'être."""
    if abs(v) in NIVEAUX:
        return "niveau de confiance"
    if abs(v) in ENUM:
        return "entier d'énumération"
    if v == int(v) and int(v) in ANNEES:
        return "millésime"
    return None


def nombres(txt, latex=True):
    """Les nombres d'un fragment qui sont des résultats, donc à confirmer."""
    return [v for v in extrait(txt, latex) if exemption(v) is None]


def charge_sorties(dossier):
    src = {}
    for f in os.listdir(dossier):
        if f.endswith(".txt"):
            with open(os.path.join(dossier, f), encoding="utf-8", errors="replace") as fh:
                src[f[:-4]] = nombres(fh.read(), latex=False)
    return src


def decimales(v):
    """Nombre de décimales significatives de v, tel qu'il a été écrit dans le mémoire."""
    s = repr(float(v))
    if "e" in s or "E" in s:
        return 0
    return len(s.split(".")[1].rstrip("0")) if "." in s else 0


def confirme(v, pool):
    """Tolérance d'arrondi : 0,6 % en relatif, ou la demi-unité du dernier chiffre écrit.

    POURQUOI PAS UN PLANCHER ABSOLU FIXE. La version precedente prenait max(0,5 ; 0,6 %),
    et ce plancher de 0,5 etait ecrasant sur les petites valeurs : il rendait 0,9 apparie
    a tout ce qui tombe entre 0,4 et 1,4, donc confirme par un 0,99 sans rapport. Tout
    nombre publie sous 83 etait apparie plus largement que la tolerance relative annoncee.

    LA BONNE BORNE EST CELLE DE L'ARRONDI D'ECRITURE. Un nombre ecrit « 122 » vient d'une
    valeur dans [121,5 ; 122,5[ : la tolerance est 0,5. Ecrit « 2,1 », il vient de
    [2,05 ; 2,15[ : la tolerance est 0,05. Ecrit « 0,90 », elle est 0,005. On prend donc la
    demi-unite du dernier chiffre ECRIT, et l'on garde la tolerance relative de 0,6 % pour
    les grands nombres, ou le mémoire arrondit plus librement que le dernier chiffre.
    """
    tol = max(0.006 * abs(v), 0.5 * 10 ** (-decimales(v)))
    return any(abs(v - w) <= tol for w in pool)


def main():
    dossier = sys.argv[1] if len(sys.argv) > 1 else None
    chapitre = sys.argv[2] if len(sys.argv) > 2 else CHAPITRE_DEFAUT
    if not dossier or not os.path.isdir(dossier):
        print(__doc__)
        sys.exit(1)

    sorties = charge_sorties(dossier)
    print(f"sorties chargées : {', '.join(sorted(sorties))}")
    with open(chapitre, encoding="utf-8") as f:
        brut = f.read()
    lignes = brut.split("\n")

    # HORS SCRIPT PAR NATURE, ET DECLARE COMME TEL. Un chapitre de demonstrations ou de
    # notations ne peut citer aucun script : ses nombres sont des constantes d'enonce, pas des
    # resultats. Les laisser dans le meme sac que les sections qu'on a simplement oublie de
    # rattacher est trompeur dans les deux sens : cela gonfle le hors-controle, et cela permet
    # d'y cacher une vraie omission. La declaration se fait dans le chapitre lui-meme, par un
    # commentaire LaTeX, avec sa justification, et le rapport la compte a part.
    m_decl = re.search(r"%\s*HARNAIS-HORS-SCRIPT\s*:\s*(.+)", brut)
    declare = m_decl.group(1).strip() if m_decl else None

    # Découpage en sections. On coupe aussi sur \section* et \subsection(*) : deux
    # chapitres (données, socle) n'ont qu'une seule \section numérotée sur une dizaine de
    # pages et structurent le reste en sous-sections étoilées. Sans cela leur texte tombe
    # entier dans un seul bloc, dont le pool de scripts est l'union de tout ce qui y est
    # cité : le taux de confirmation s'effondre pour une raison de découpage, pas de fond.
    sections, cur, titre = [], [], "(préambule de chapitre)"
    coupe = re.compile(r"\\(?:sub)?section\*?\{(.+?)\}")
    for l in lignes:
        m = coupe.match(l.strip())
        if m:
            sections.append((titre, cur))
            titre, cur = m.group(1), []
        else:
            cur.append(l)
    sections.append((titre, cur))

    tot = ok = orphelins = 0
    sans_source = []
    sections_declarees = []
    detail = []
    exemptes = {}
    for titre, corps in sections:
        txt = "\n".join(corps)
        # Tout \texttt{...} dont le contenu est un identifiant de script connu. Plus
        # robuste que d'exiger le mot "script" juste avant : les sections citent souvent
        # en liste (\og scripts 16b, 20b \fg), et le second serait alors manqué.
        # LE NOM LONG COMPTE AUSSI. Une section citait « script \texttt{32\_proprietes\_formelles} »
        # et le harnais la classait sans source, donc hors controle : le motif n'acceptait que
        # l'identifiant nu. La citation longue est pourtant la meilleure des deux pour un lecteur.
        # On capture donc le prefixe numerique, avec ou sans suite.
        # ET LA CITATION PEUT ETRE INVISIBLE, DEPUIS LE 15 SEPTEMBRE 2026. Kelian a
        # demande que le memoire ne porte plus de renvois << scripts 43 et 68 >> :
        # un memoire publie par l'Institut n'en montre pas, c'est un appareil de
        # tracabilite interne et il a l'air de notes de laboratoire. Le rattachement
        # SUBSISTE donc, mais dans un commentaire LaTeX, qui ne s'imprime pas :
        #     % SOURCES-SCRIPTS: 43 68
        # Sans ce second canal, retirer les lignes visibles ferait tomber la
        # couverture de 100 % a zero et le dispositif de verification entier avec.
        cites = sorted({c for c in re.findall(r"\\texttt\{([0-9]+[a-z]?)(?:\\?_[^}]*)?\}", txt)
                        if c in sorties}
                       | {c for ligne in re.findall(r"%\s*SOURCES-SCRIPTS\s*:\s*(.+)", txt)
                          for c in re.findall(r"[0-9]+[a-z]?", ligne) if c in sorties})
        bruts = extrait(txt)
        vals = [v for v in bruts if exemption(v) is None]
        if not vals:
            continue
        # Meme mecanisme que la declaration de chapitre, a l'echelle d'une section : le chapitre
        # socle est verifie de bout en bout sauf sa section de theoremes, dont les nombres sont
        # des constantes d'enonce. Declarer la section vaut mieux que declarer le chapitre.
        m_sec = re.search(r"%\s*HARNAIS-HORS-SECTION\s*:\s*(.+)", txt)
        if m_sec and not cites:
            sections_declarees.append((titre, len(vals), m_sec.group(1).strip()))
            continue
        if not cites:
            sans_source.append((titre, len(vals)))
            continue
        for v in bruts:
            motif = exemption(v)
            if motif:
                exemptes[motif] = exemptes.get(motif, 0) + 1
        pool = [v for c in cites for v in sorties.get(c, [])]
        manquants = [v for v in vals if not confirme(v, pool)]
        tot += len(vals)
        ok += len(vals) - len(manquants)
        orphelins += len(manquants)
        detail.append((titre, cites, len(vals), manquants))

    print("\n" + "=" * 78)
    print("VÉRIFICATION SECTION PAR SECTION")
    print("=" * 78)
    for titre, cites, n, manquants in detail:
        etat = "OK" if not manquants else f"{len(manquants)} non confirmé(s)"
        print(f"\n  {titre[:58]:<58}")
        print(f"    scripts {','.join(cites):<14} {n:>3} nombres   -> {etat}")
        if manquants:
            aff = ", ".join(f"{v:g}" for v in manquants[:14])
            print(f"    non trouvés : {aff}" + (" ..." if len(manquants) > 14 else ""))

    if sans_source:
        print("\n  Sections SANS script cité (non vérifiables automatiquement) :")
        for t, n in sans_source:
            print(f"    {t[:56]:<56} {n:>3} nombres")

    if exemptes:
        n_ex = sum(exemptes.values())
        print(f"\n  Exemptés de vérification ({n_ex} nombres, comptés ici pour que le "
              f"dénominateur\n  du taux ne soit pas silencieusement filtré) :")
        for motif, n in sorted(exemptes.items(), key=lambda kv: -kv[1]):
            print(f"    {motif:<24} {n:>4}")

    # LA COUVERTURE, A COTE DU TAUX, ET C'EST LE CHIFFRE QUI ENGAGE. Le taux de confirmation
    # se calcule sur les seules sections qui citent un script. Une section qui n'en cite aucun
    # n'est pas confirmee : elle est HORS CONTROLE, ce qui est pire, et cela ne se voyait nulle
    # part dans le rapport. On pouvait donc faire monter le taux en retirant une citation. Les
    # cinq chiffres perimes trouves cette semaine venaient tous de la zone non couverte ou d'une
    # exemption silencieuse, jamais d'un non confirme. Le harnais imprime desormais les deux.
    if sections_declarees:
        print("\n  Sections DECLAREES hors script, avec leur motif :")
        for t, n, motif in sections_declarees:
            print(f"    {t[:52]:<52} {n:>3} nombres")
            print(f"      {motif}")

    decl_sec = sum(n for _, n, _ in sections_declarees)
    brut_hors = sum(n for _, n in sans_source)
    hors = 0 if declare else brut_hors
    publie = tot + brut_hors + decl_sec
    if publie:
        print("\n" + "=" * 78)
        print(f"  COUVERTURE : {tot} nombres sous controle sur {publie} publies "
              f"({100*tot/publie:.1f} %).")
        if decl_sec:
            print(f"  {decl_sec} nombres hors script PAR NATURE, declares section par section.")
        if declare:
            print(f"  {brut_hors} nombres hors script PAR NATURE, declare dans le chapitre :")
            print(f"    {declare}")
            print(f"  Hors controle non declare : 0. C'est ce dernier chiffre qui doit tomber a")
            print(f"  zero, pas le precedent.")
        elif hors:
            print(f"  {hors} nombres vivent dans une section qui ne cite aucun script : ils ne")
            print(f"  sont ni confirmes ni infirmes, personne ne les regarde. Voir la liste")
            print(f"  ci-dessus, qui est a relire et non a parcourir.")

    print("\n" + "=" * 78)
    print(f"  {tot} nombres vérifiables, {ok} confirmés, {orphelins} non confirmés "
          f"({100*ok/tot:.1f} % de confirmation)" if tot else "  rien à vérifier")
    print("=" * 78)
    print("  Un nombre non confirmé n'est pas forcément faux : il peut venir d'un calcul")
    print("  intermédiaire non imprimé, d'un arrondi de rédaction, ou d'une autre source.")
    print("  Mais chacun doit être justifié à la main. C'est la liste de travail.")


if __name__ == "__main__":
    main()
