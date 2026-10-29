"""Restitution du calcul 129 : tables et figure, sans recalibration du mémoire."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def nombre(x, d=3):
    return f'{x:.{d}f}'.replace('.', '{,}')


def intervalle(xs, d=3):
    return '[' + ' ; '.join(nombre(x, d) for x in xs) + ']'


def main():
    r = json.loads((ROOT/'resultats/institut_approfondissements/129.json').read_text(encoding='utf-8'))
    assert (r['statut'], r['n'], r['n_exces'], len(r['echecs'])) == ('termine', 570, 86, 0)
    assert all(v['succes'] == r['B_estimateurs'] for v in r['estimateurs'].values())
    assert all(v['succes'] == r['B_gof'] for v in r['balayage'])
    rows = r['estimateurs']
    court = ['% Généré depuis 129.json par outils/ci/actualiser_complements_hugo.py.',
             r'\small\textbf{Indice et IC bootstrap à 95 \%}\\[2mm]',
             r'\begin{tabular}{@{}lcc@{}}\toprule Estimateur & $\hat\xi$ & IC 95 \% \\\midrule']
    complet = [court[0], r'\scriptsize\renewcommand{\arraystretch}{1.45}',
        r'\begin{tabularx}{\textwidth}{Yrrrrrr}\toprule',
        r'Estimateur & $\hat\xi$ & IC 95 \% & $\hat\sigma$ (M€) & IC 95 \% & $q_{99,5\%}$ (M€) & IC 95 \% \\\midrule']
    memoire = [court[0], '% SOURCES-SCRIPTS: 129', r'\begin{center}\small',
        r'\begin{tabular}{lcc}\toprule Estimateur & $\hat\xi$ & IC bootstrap à $95\,\%$ \\\midrule']
    for nom, row in rows.items():
        texte = f"{nom} & {nombre(row['xi'])} & {intervalle(row['xi_ic95'])}"
        court.append(texte+r' \\')
        memoire.append(texte+r' \\')
        suite = ' & -- & -- & -- & --'
        if row['sigma'] is not None:
            suite = f" & {nombre(row['sigma'],2)} & {intervalle(row['sigma_ic95'],1)} & {nombre(row['q'],0)} & {intervalle(row['q_ic95'],0)}"
        complet.append(texte+suite+r' \\')
    court += [r'\bottomrule\end{tabular}\\[2mm]',
              r'\footnotesize Seuil fixé : 20,03 M€.\\',
              r'Les intervalles portent sur les excès observés ; ils n’intègrent pas le choix du seuil.']
    complet += [r'\bottomrule\end{tabularx}']
    memoire += [r'\bottomrule\end{tabular}\end{center}']
    for fichier, lignes in [('recherche/slides/beamer/table129_resume.tex',court),
                            ('recherche/slides/beamer/table129_complete.tex',complet),
                            ('memoire/chapitres_inst/complements_hugo_table.tex',memoire)]:
        (ROOT/fichier).write_text('\n'.join(lignes)+'\n',encoding='utf-8')
    plt.rcParams.update({'font.size':12, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, ax = plt.subplots(figsize=(6.4,3.7))
    for cle, label, col in [('p_ks','KS','#223e55'),('p_ad','AD','#b10031')]:
        ax.plot([v['u'] for v in r['balayage']], [v[cle] for v in r['balayage']], 'o-', label=label, color=col)
    ax.axhline(.05, ls=':', color='grey', label='Seuil de test : 5 %')
    ax.axvline(r['u'], ls='--', lw=1, color='grey')
    ax.text(r['u']+2,.93,'20,03 M€',fontsize=10)
    ax.set(xlabel='Seuil monétaire u (M€)', ylabel='p-value après réestimation', ylim=(0,1.02))
    ax.legend(loc='lower right',fontsize=10); ax.grid(alpha=.14); fig.tight_layout()
    fig.savefig(ROOT/'recherche/vasicek_lab/figures_inst/G129_tests_seuils.png',dpi=200); plt.close(fig)
    sortie = ROOT/'resultats/sorties_verif_inst/129.txt'
    # Compléter le journal existant avec les paramètres enregistrés lors du calcul,
    # sans relancer ni prétendre relancer les simulations.
    protocole = f"Protocole sauvegardé dans 129.json : B_gof={r['B_gof']} ; B_estimateurs={r['B_estimateurs']} ; graine={r['graine']}."
    journal = sortie.read_text(encoding='utf-8')
    if protocole not in journal:
        sortie.write_text(journal.rstrip()+'\n\nRestitution des métadonnées du calcul, sans nouvelle simulation :\n'+protocole+'\n', encoding='utf-8')
    contenu = sortie.read_bytes()
    ligne = f"| 129 | 129_seuils_hugo_institut.py | 0 | non | {r['duree_secondes']:.0f} s | {len(contenu)} | {hashlib.md5(contenu).hexdigest()[:8]} |"
    manifeste = ROOT/'resultats/sorties_verif_inst/MANIFESTE.md'
    texte = manifeste.read_text(encoding='utf-8')
    texte = '\n'.join(l for l in texte.splitlines() if not l.startswith('| 129 |'))+'\n'+ligne+'\n'
    manifeste.write_text(texte,encoding='utf-8')
    print('Tables, figure et manifeste du complément 129 actualisés.')


if __name__ == '__main__':
    main()
