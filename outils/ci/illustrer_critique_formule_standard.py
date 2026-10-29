"""Figure de lecture de la critique de la Formule Standard (chapitre 1), sans valeur estimée.

(a) La charge de risque opérationnel est une fonction du volume, plafonnée à 0,3 BSCR :
    deux entités de même volume reçoivent la même charge quel que soit leur état.
(b) Le besoin ORSA est le quantile à 99,5 % d'une loi de perte que l'état déplace.
Les lois du panneau (b) sont des log-normales d'illustration, choisies pour la lisibilité ;
elles ne reprennent aucun paramètre calibré du mémoire. Mêmes police et couleurs que
illustrer_methode_institut.py.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'memoire/figures_pedagogie'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                     'font.size': 10, 'pdf.fonttype': 42, 'mathtext.fontset': 'dejavuserif'})
INK = '#25333F'
RED = '#A52D39'
TEAL = '#14828B'
GRIS = '#81878C'

fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 3.25), gridspec_kw={'wspace': .28})

# (a) Charge forfaitaire : linéaire en volume, puis plafonnée.
v = np.linspace(0, 10, 200)
plafond = 6.0
a.plot(v, np.minimum(.9 * v, plafond), color=INK, linewidth=1.6)
a.axhline(plafond, color=GRIS, linewidth=.8, linestyle='--')
a.text(9.8, plafond - 1.25, 'plafond $0{,}3\\,BSCR$\n($BSCR$ fixé)', color=GRIS, fontsize=9, ha='right')
v0 = 4.4
a.plot([v0], [.9 * v0], 'o', markersize=11, markerfacecolor='none', markeredgecolor=RED, markeredgewidth=1.5)
a.plot([v0], [.9 * v0], 'o', markersize=5.5, color=TEAL)
a.annotate('C et NC :\nmême charge', xy=(v0, .9 * v0), xytext=(5.6, 1.5), fontsize=9.5,
           arrowprops={'arrowstyle': '-|>', 'color': INK, 'linewidth': .9})
a.set_xlabel('Volume (primes, provisions)')
a.set_ylabel(r'$SCR_{Op}$')
a.set_title('(a) Formule Standard', fontsize=10.5, color=INK, loc='left')

# (b) Besoin ORSA : quantile d'une loi de perte déplacée par l'état.
x = np.linspace(.01, 12, 600)
lois = [('Conforme', TEAL, stats.lognorm(s=.45, scale=1.5)),
        ('Non conforme', RED, stats.lognorm(s=.5, scale=2.4))]
for nom, couleur, loi in lois:
    q = loi.ppf(.995)
    b.plot(x, loi.pdf(x), color=couleur, linewidth=1.5, label=nom)
    b.fill_between(x, loi.pdf(x), where=x >= q, color=couleur, alpha=.25)
    b.axvline(q, color=couleur, linewidth=.9, linestyle='--')
b.text(lois[0][2].ppf(.995) + .15, .42, r'$B_C$', color=TEAL, ha='left', fontsize=11)
b.text(lois[1][2].ppf(.995) + .15, .42, r'$B_{NC}$', color=RED, ha='left', fontsize=11)
b.set_xlabel(r'Perte annuelle $L$')
b.set_ylabel('Densité')
b.set_xlim(0, 12)
b.legend(frameon=False, fontsize=9, loc='upper right')
b.set_title(r'(b) Besoin ORSA : $\mathrm{VaR}_{99{,}5\,\%}(L\mid e)$', fontsize=10.5, color=INK, loc='left')

for ax in (a, b):
    ax.set_xticks([]); ax.set_yticks([])
    for cote in ('top', 'right'):
        ax.spines[cote].set_visible(False)
    for cote in ('left', 'bottom'):
        ax.spines[cote].set_color(INK)

fig.subplots_adjust(left=.05, right=.99, top=.9, bottom=.12)
fig.savefig(OUT / 'critique_formule_standard.pdf')
plt.close(fig)
print('Figure écrite :', OUT / 'critique_formule_standard.pdf')
