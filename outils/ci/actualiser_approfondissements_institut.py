#!/usr/bin/env python3
"""Produit les tableaux Institut et rattache les sorties agrégées 119/120 au manifeste."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
RESULTATS = ROOT/'resultats/institut_approfondissements'
CHAPITRES = ROOT/'memoire/chapitres_inst'
SORTIES = ROOT/'resultats/sorties_verif_inst'


def entier(x):
    return f'{x:,.0f}'.replace(',', r'\,')


def decimal(x, digits=2):
    return f'{x:.{digits}f}'.replace('.', '{,}')


def ecrire(nom, texte):
    (CHAPITRES/nom).write_text(texte, encoding='utf-8')


def main():
    a = json.loads((RESULTATS/'119.json').read_text(encoding='utf-8'))
    b = json.loads((RESULTATS/'120.json').read_text(encoding='utf-8'))
    if b.get('statut') != 'termine' or len(b['replications']) != b['replications_prevues']:
        raise ValueError('Bootstrap incomplet ou échecs à examiner avant intégration')
    noms = dict(additive='Additive', moyenne='Moyenne', unique='Unique', amplifiee='Amplifiée')
    texte = r'''% Généré par actualiser_approfondissements_institut.py depuis 119.json.
% SOURCES-SCRIPTS: 119
\begin{center}\footnotesize
\renewcommand{\arraystretch}{1.15}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
\textbf{Perte} & \textbf{Conforme} & \textbf{Non conforme} & \textbf{Écart} & \textbf{Rapport} & \textbf{Interaction} \\
\midrule
'''
    for regle, v in a['resultats'].items():
        texte += (noms[regle]+' & '+ ' & '.join('$'+entier(v[k])+'$' for k in ('conforme', 'non_conforme', 'ecart'))
                  +' & $'+decimal(v['rapport'])+'$ & $'+entier(v['interaction'])+r'$ \\'+'\n')
    texte += r'''\bottomrule
\end{tabular}
\end{center}
\noindent Montants en M\euro{} pour le périmètre financier étudié. Les quatre règles
utilisent le calage historique ; l'incident amplifié a un exposant posé égal à un.
Les déplacements sous raffinement de grille restent de l'ordre du M\euro{}.
'''
    ecrire('approfondissements_table_unite.tex', texte)
    c = b['calibration_locale']
    texte = '% Généré depuis 120.json ; intervalles conditionnels, non incertitude totale.\n% SOURCES-SCRIPTS: 120\n'
    texte += (f"L'échantillon compte ${entier(b['n'])}$ pertes et ${entier(b['n_exces'])}$ excès. "
              f"L'ajustement donne $\\hat\\xi={decimal(c['xi'], 5)}$, "
              f"$\\hat\\sigma={decimal(c['sigma'], 2)}$~M\\euro{{}} et "
              f"$\\hat p_u={decimal(c['pu'], 6)}$. "
              f"Nous retenons ${entier(len(b['replications']))}$ réplications, sans échec.\n\n")
    texte += r'''\begin{center}\footnotesize
\renewcommand{\arraystretch}{1.15}
\begin{tabular}{@{}lrrrr@{}}
\toprule
\textbf{Perte} & \multicolumn{2}{c}{\textbf{Écart (M\euro)}} & \multicolumn{2}{c}{\textbf{Rapport}} \\
 & \textbf{Ajustement} & \textbf{Bande à 95\,\%} & \textbf{Ajustement} & \textbf{Bande à 95\,\%} \\
\midrule
'''
    for regle, r in b['resume'].items():
        e, q, central = r['ecart'], r['rapport'], b['central'][regle]
        texte += (noms[regle]+' & $'+entier(central['ecart'])+'$ & $['+entier(e['p025'])+r'\,;'
                  +entier(e['p975'])+']$ & $'+decimal(central['rapport'])+'$ & $['
                  +decimal(q['p025'])+r'\,;'+decimal(q['p975'])+r']$ \\'+'\n')
    texte += r'''\bottomrule
\end{tabular}
\end{center}
'''
    texte += (f"\\noindent Bandes percentiles nominales à $95\\,\\%$, calculées directement sur les paires. "
              f"La corrélation bootstrap entre $\\xi$ et $\\sigma$ vaut "
              f"${decimal(b['correlation_parametres'][0][1], 3)}$. "
              f"Les ${entier(b['ajustements_xi_superieur_egal_un'])}$ ajustements avec $\\xi\\geq1$ "
              "sont conservés : leur moyenne est infinie, leur quantile reste défini. "
              "Le raffinement numérique des ajustements extrêmes et d'un ajustement médian "
              "ne déplace aucun capital de plus de "
              f"${decimal(100*max(r['ecart_relatif_max'] for r in b['controles_numeriques']), 2)}\\,\\%$.\n")
    ecrire('approfondissements_table_bootstrap.tex', texte)
    m = json.loads((RESULTATS/'123.json').read_text(encoding='utf-8'))
    texte = ('% Généré depuis 123.json ; mélange empirique, conditionnel aux hypothèses du moteur.\n'
             '% SOURCES-SCRIPTS: 123\n'
             '\\begin{center}\\footnotesize\n\\renewcommand{\\arraystretch}{1.15}\n'
             '\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n'
             '\\textbf{Perte} & \\multicolumn{2}{c}{\\textbf{Conforme (M\\euro)}} & '
             '\\multicolumn{2}{c}{\\textbf{Non conforme (M\\euro)}} & \\multicolumn{2}{c}{\\textbf{Rapport}} \\\\\n'
             ' & \\textbf{Central} & \\textbf{Mélange} & \\textbf{Central} & \\textbf{Mélange} '
             '& \\textbf{Central} & \\textbf{Mélange} \\\\\n\\midrule\n')
    for regle, v in m['resultats'].items():
        c_, n_ = v['conforme'], v['non_conforme']
        texte += (noms[regle]+' & $'+entier(c_['central'])+'$ & $'+entier(c_['melange']['0.995'])+'$ & $'
                  +entier(n_['central'])+'$ & $'+entier(n_['melange']['0.995'])+'$ & $'
                  +decimal(n_['central']/c_['central'])+'$ & $'+decimal(v['contraste_melange']['rapport'])+r'$ \\'+'\n')
    texte += ('\\bottomrule\n\\end{tabular}\n\\end{center}\n'
              '\\noindent Quantiles annuels à $99{,}5\\,\\%$, périmètre financier étudié. Le mélange moyenne les lois annuelles '
              f"de ${entier(m['n_calibrations'])}$ calibrations ; la lecture se fait sur une grille de résolution relative "
              f"${decimal(100*m['resolution_relative_grille'])}\\,\\%$.\n")
    ecrire('approfondissements_table_melange.tex', texte)
    t124 = json.loads((RESULTATS/'124.json').read_text(encoding='utf-8'))
    noms124 = dict(S='Stationnaire', D='Échelle variable', R='Deux régimes',
                   P='Taux en tendance', DP='Échelle variable et taux en tendance')
    texte = ('% Généré depuis 124.json ; validation chronologique, capital pour l\'année suivante.\n'
             '% SOURCES-SCRIPTS: 124\n'
             '\\begin{center}\\footnotesize\n\\renewcommand{\\arraystretch}{1.15}\n'
             '\\begin{tabular}{@{}lrrr@{}}\n\\toprule\n'
             '\\textbf{Loi retenue} & \\textbf{Conforme} & \\textbf{Non conforme} & \\textbf{Rapport} \\\\\n\\midrule\n')
    for cle, nom in noms124.items():
        v = t124['capitaux'][cle]['additive']
        texte += (nom+' & $'+entier(v['conforme'])+'$ & $'+entier(v['non_conforme'])+'$ & $'
                  +decimal(v['rapport'])+r'$ \\'+'\n')
    texte += ('\\bottomrule\n\\end{tabular}\n\\end{center}\n'
              f"\\noindent Capital annuel à $99{{,}}5\\,\\%$ pour {t124['annee_prevue']} (M\\euro, règle additive, "
              f"périmètre financier étudié), après ajustement sur 2000--{t124['annees_notees'][1]}.\n")
    ecrire('approfondissements_table_temporelle.tex', texte)
    t125 = json.loads((RESULTATS/'125.json').read_text(encoding='utf-8'))
    texte = ('% Généré depuis 125.json ; rééchantillonnage annuel conjoint, bandes percentiles exploratoires.\n'
             '% SOURCES-SCRIPTS: 125\n'
             '\\begin{center}\\footnotesize\n\\renewcommand{\\arraystretch}{1.15}\n'
             '\\begin{tabular}{@{}llrrr@{}}\n\\toprule\n'
             '\\textbf{Tirage} & \\textbf{Fréquence} & \\textbf{Rapport médian} & \\textbf{Bande à 90\\,\\%} '
             '& \\textbf{Non conforme, bande à 90\\,\\%} \\\\\n\\midrule\n')
    for schema, nom_s in (('annees', 'Années'), ('blocs', 'Blocs de trois années')):
        for variante, nom_f in (('sans', 'fixée'), ('avec', 'propagée')):
            v = t125['schemas'][schema]['regles']['additive'][variante]
            texte += (f"{nom_s} & {nom_f} & ${decimal(v['rapport']['p50'])}$ & "
                      f"$[{decimal(v['rapport']['p5'])}\\,;{decimal(v['rapport']['p95'])}]$ & "
                      f"$[{entier(v['non_conforme']['p5'])}\\,;{entier(v['non_conforme']['p95'])}]$ \\\\\n")
    texte += ('\\bottomrule\n\\end{tabular}\n\\end{center}\n'
              f"\\noindent Règle additive, périmètre financier étudié, {t125['schemas']['annees']['tirages']} tirages par schéma "
              "(capital en M\\euro).\n")
    ecrire('approfondissements_table_annuel.tex', texte)
    manifeste = SORTIES/'MANIFESTE.md'
    texte = manifeste.read_text(encoding='utf-8')
    commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, text=True).strip()
    stamp = datetime.datetime.now().strftime('%d/%m/%Y a %H:%M')
    texte = re.sub(r'Dernier lancement le [^\n]+, code du commit `[^`]+`',
                   f'Dernier lancement le {stamp}, code du commit `{commit}`', texte)
    m123 = json.loads((RESULTATS/'123.json').read_text(encoding='utf-8'))
    m124 = json.loads((RESULTATS/'124.json').read_text(encoding='utf-8'))
    m125 = json.loads((RESULTATS/'125.json').read_text(encoding='utf-8'))
    m126 = json.loads((RESULTATS/'126.json').read_text(encoding='utf-8'))
    m127 = json.loads((RESULTATS/'127.json').read_text(encoding='utf-8'))
    m128 = json.loads((RESULTATS/'128.json').read_text(encoding='utf-8'))
    for numero, nom, objet in [(119, '119_unite_perte_capital_institut.py', a),
                              (120, '120_bootstrap_joint_institut.py', b),
                              (123, '123_melange_predictif_institut.py', m123),
                              (124, '124_validation_temporelle_institut.py', m124),
                              (125, '125_reechantillonnage_annuel_institut.py', m125),
                              (126, '126_dependance_couts_incident_institut.py', m126),
                              (127, '127_extrema_hors_sommets_institut.py', m127),
                              (128, '128_residus_compenses_comptage_institut.py', m128)]:
        brut = (SORTIES/f'{numero}.txt').read_bytes()
        if not brut.strip() or b'Traceback' in brut:
            raise ValueError(f'Sortie {numero} absente ou en erreur')
        ligne = (f'| {numero} | {nom} | 0 | non | {int(objet["duree_secondes"])} s | '
                 f'{len(brut)} | {hashlib.md5(brut).hexdigest()[:8]} |')
        motif = rf'^\| {numero} \|.*$'
        texte = re.sub(motif, ligne, texte, flags=re.M) if re.search(motif, texte, flags=re.M) else texte.rstrip()+'\n'+ligne+'\n'
    if 'Les expériences 119 et 120' not in texte:
        texte = texte.replace('| script | fichier |',
            "Les expériences 119 et 120 sont propres à la reprise Institut : le calage historique\n"
            "reste inchangé et le bootstrap réestime conjointement la calibration dans un moteur séparé.\n\n"
            '| script | fichier |')
    manifeste.write_text(texte, encoding='utf-8')
    print('Deux tableaux Institut et manifeste actualisés ; données individuelles non lues.')


if __name__ == '__main__':
    main()
