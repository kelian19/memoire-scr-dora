#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
99 : d'ou vient le passage de 21,6 a 53,6 incidents par an entre l'etat conforme et l'etat
non conforme.

POURQUOI CE SCRIPT EXISTE. Le canal de frequence est le levier qui commande le chiffre de tete
(la frequence seule porte le besoin de capital a 10 377 M, script 92), et le memoire ne donnait
nulle part la provenance de son multiplicateur : 53,6 n'apparaissait que dans la note de synthese.
Le calcul vit dans src.frequency.negbin.compute_lambda_scenario, appele par
canaux_conformite.LAM_NC. Ce script l'imprime terme a terme :
  lambda_NC = lambda_C * somme_v pi_v * m_v,
  pi_v : part du vecteur d'attaque v dans Hackmageddon (S1 2026, 1 041 incidents, citation
         externe non recalculable, script 63) ;
  m_v  : centre de la fourchette de multiplicateur du scenario S2 (non conforme) pour le vecteur v,
         fourchette POSEE et ancree sur un effet publie (Ponemon/IBM, ENISA, IBM/SecurityScorecard,
         Microsoft Research), jamais estimee sur donnee du projet.

CONTROLE. Le produit recompose ici doit redonner canaux_conformite.LAM_NC a 1e-9 pres, sans quoi
le memoire documenterait un autre nombre que celui du moteur.

Aucune simulation, aucune donnee sous licence : le script tourne sur les deux postes.
"""

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DEPOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, ROOT_DEPOT):
    if p not in sys.path:
        sys.path.insert(0, p)
import canaux_conformite as cx                                            # noqa: E402
from src.frequency.negbin import (HACKMAGEDDON_PROPORTIONS,               # noqa: E402
                                  MULTIPLICATEURS_DORA, SOURCES_MULTIPLICATEURS)

WID = 88


def titre(s):
    print("\n" + "=" * WID + f"\n{s}\n" + "=" * WID)


# =====================================================================================
titre("1. Le multiplicateur de frequence, vecteur par vecteur")
# =====================================================================================
lam_c = cx.LAM_C
print(f"  lambda a l'etat conforme (S0, OpRisk) : {lam_c:.2f} incidents materiels par an")
print(f"\n  {'vecteur':<22}{'part pi_v':>10}{'S1 partiel':>14}{'S2 non conf.':>15}"
      f"{'centre S2':>11}{'pi_v x m_v':>12}")
agg = {"S1_partiel": 0.0, "S2_non_conforme": 0.0}
for v, pi in HACKMAGEDDON_PROPORTIONS.items():
    lo1, hi1 = MULTIPLICATEURS_DORA["S1_partiel"][v]
    lo2, hi2 = MULTIPLICATEURS_DORA["S2_non_conforme"][v]
    m1, m2 = (lo1 + hi1) / 2, (lo2 + hi2) / 2
    agg["S1_partiel"] += pi * m1
    agg["S2_non_conforme"] += pi * m2
    print(f"  {v:<22}{pi:>10.3f}{'[' + format(lo1, '.1f') + ' ; ' + format(hi1, '.1f') + ']':>14}"
          f"{'[' + format(lo2, '.1f') + ' ; ' + format(hi2, '.1f') + ']':>15}{m2:>11.2f}{pi*m2:>12.4f}")
print(f"  {'somme':<22}{sum(HACKMAGEDDON_PROPORTIONS.values()):>10.3f}{'':>40}"
      f"{agg['S2_non_conforme']:>12.4f}")

m_nc = agg["S2_non_conforme"]
m_pc = agg["S1_partiel"]
lam_nc = lam_c * m_nc
print(f"\n  multiplicateur agrege, etat partiellement conforme (S1) : {m_pc:.4f}")
print(f"  multiplicateur agrege, etat non conforme (S2)           : {m_nc:.4f}")
print(f"  lambda non conforme = {lam_c:.2f} x {m_nc:.4f} = {lam_nc:.2f} incidents par an")

ecart = abs(lam_nc - cx.LAM_NC)
assert ecart < 1e-9, f"le produit recompose s'ecarte du moteur de {ecart:.2e}"
print(f"  controle : ecart au moteur (canaux_conformite.LAM_NC) = {ecart:.1e} -> OK")

# =====================================================================================
titre("2. Ce qui est ancre et ce qui est pose")
# =====================================================================================
print("  Les PARTS pi_v sont une citation externe (Hackmageddon, non recalculable). Les BORNES des")
print("  fourchettes sont POSEES : chacune traduit un effet publie en multiplicateur prudent, sans")
print("  estimation sur donnee du projet. Le multiplicateur est donc semi-ancre, au sens de la")
print("  table des parametres : sa provenance est tracee, sa valeur n'est pas calibree.")
for v, src in SOURCES_MULTIPLICATEURS.items():
    print(f"  {v:<22} {src}")

part_ident = HACKMAGEDDON_PROPORTIONS["identifiants"] * \
    sum(MULTIPLICATEURS_DORA["S2_non_conforme"]["identifiants"]) / 2
print(f"\n  poids du seul vecteur 'identifiants' dans le multiplicateur S2 : {part_ident:.4f}, soit "
      f"{100*part_ident/m_nc:.1f} % du total pour {100*HACKMAGEDDON_PROPORTIONS['identifiants']:.1f} % "
      "des incidents")
lo_tot = sum(pi * MULTIPLICATEURS_DORA["S2_non_conforme"][v][0] for v, pi in HACKMAGEDDON_PROPORTIONS.items())
hi_tot = sum(pi * MULTIPLICATEURS_DORA["S2_non_conforme"][v][1] for v, pi in HACKMAGEDDON_PROPORTIONS.items())
print(f"  multiplicateur S2 aux bornes basses : {lo_tot:.4f} ; aux bornes hautes : {hi_tot:.4f}")
print(f"  lambda non conforme correspondant : {lam_c*lo_tot:.2f} a {lam_c*hi_tot:.2f} incidents par an")

# =====================================================================================
titre("3. Grandeurs citees, sans separateur")
# =====================================================================================
print(f"  lambda conforme {lam_c:.2f} ; multiplicateur S2 {m_nc:.3f} ; lambda non conforme {lam_nc:.2f}")
print(f"  multiplicateur S1 {m_pc:.3f} ; lambda partiel {lam_c*m_pc:.2f}")
print(f"  plage S2 : multiplicateur {lo_tot:.3f} a {hi_tot:.3f}, lambda {lam_c*lo_tot:.1f} a {lam_c*hi_tot:.1f}")
print(f"  vecteur identifiants : {100*part_ident/m_nc:.1f} % du multiplicateur S2")
