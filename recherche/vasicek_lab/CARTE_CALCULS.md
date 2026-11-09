# Du mémoire au calcul

Référence : [mémoire Institut courant](../../memoire/main_institut_150.pdf), 158 pages,
version de travail du 9 novembre 2026. Le tableau « Quel chiffre pour quel calcul », au début
du chapitre 9, distingue déjà les calages ; cette page donne leurs entrées exécutables.

**Trois questions structurent la lecture.** Que change un état supposé de conformité ?
Que change une direction de propagation inconnue ? Que peut-on transposer à une entité ?
Les trois calculs ont des périmètres et des moteurs différents. Une plage obtenue dans l'un
ne doit pas être appliquée au point central d'un autre.

## Résultats courants et sensibilités

« Courant » désigne le protocole retenu dans le manuscrit, sans validation empirique de toutes
ses hypothèses. « Sensibilité » désigne une variation de convention ou de paramètre.
Les scripts qui ajustent les bases restreintes nécessitent ces bases ; lire leur sortie publiée
et contrôler le harnais ne revient pas à les réexécuter.

| Question et statut | Protocole et paramètres | Producteur et sortie | Résultat à lire et limite |
|---|---|---|---|
| **Écart entre états — courant** | Marche auto-évitante ; quatre canaux ; loi de pertes commune ; états posés. | [`43`](9_cas_usage/43_kpi_pilotables.py), moteur [`canaux_conformite.py`](canaux_conformite.py) ; [sortie 43](../../resultats/sorties_verif/43.txt). | 6 049 → 20 188 M€, ×3,34 sur le périmètre du registre. Comparaison de scénarios, sans effet causal mesuré. |
| **Choix des paramètres — courant** | Base de fréquence et queue ajustées ; parts d'attaque externes ; multiplicateurs, détection, gains et accumulation posés. | [`99`](5_etats/99_multiplicateur_frequence.py) ; [sortie 99](../../resultats/sorties_verif/99.txt). Conventions dans [`canaux_conformite.py`](canaux_conformite.py). | Le multiplicateur 2,4875 est construit à partir de fourchettes de l'auteur, non estimé entre états observés. |
| **Interactions et remédiation — sensibilité** | Corriger un canal depuis un état de départ donné ; comparer les effets isolés et conjoints. | [`68`](5_etats/68_interaction_canaux.py) ; [sortie 68](../../resultats/sorties_verif/68.txt). | Les effets isolés ne s'additionnent pas en capital. La graine commune ne supprime pas tout bruit quand la fréquence change. |
| **Direction inconnue — courant** | Cascade indépendante ; fréquence conforme ; sans détection ni accumulation ; S fixée par le barème de l'auteur. | [`30`](6_contagion/30_identification_partielle_W.py), noyau [`partial_id.py`](partial_id.py) ; [sortie 30](../../resultats/sorties_verif/30.txt). | 6 858–8 697 M€ aux 1 024 sommets. Ni extrema globaux démontrés, ni intervalle de confiance ; S n'est pas identifiée. |
| **Direction avec quatre canaux — sensibilité** | Cascade indépendante, distincte de la marche du résultat principal ; même direction dans les deux états. | [`140`](5_etats/140_diagnostics_audit_institut.py) ; [sortie 140](../../resultats/sorties_verif/140.txt). | Point ×3,28 ; facteurs aux sommets 2,96–3,51. Ne pas appliquer cette plage au point ×3,34 comme si le moteur était identique. |
| **Coût par incident — sensibilité majeure** | Fourier ; addition des coûts des piliers contre une perte unique par incident ; mêmes autres conventions. | [`119`](8_benchmarks/119_unite_perte_capital_institut.py) ; [sortie Institut 119](../../resultats/sorties_verif_inst/119.txt). | Facteur 3,35 contre 2,25. Ce contraste porte sur l'unité économique de perte, pas seulement sur la méthode numérique. |
| **Incertitude de calibration — sensibilité** | Bootstrap conjoint de la sévérité ; autres hypothèses conditionnantes maintenues. | [`120`](1_fondations/120_bootstrap_joint_institut.py) ; [sortie Institut 120](../../resultats/sorties_verif_inst/120.txt). | La bande ne couvre pas toutes les erreurs de modèle ni les paramètres posés. |
| **Passage à l'entité — courant** | Panel de grandes institutions ; fréquence liée à la taille ; corps de pertes positif ; bootstrap par firme ; seul g au niveau non conforme. | [`136`](2_donnees/136_descente_grandes_institutions.py), module [`descente.py`](descente.py) ; [sortie 136](../../resultats/sorties_verif_inst/136.txt). | 24,1 M€ pour l'entité notionnelle de 20 Md$ d'actifs. Extrapolation conditionnelle ; effet du métier non retenu. |
| **Sélection et enregistrement — sensibilité majeure** | Diagnostics de descente et scénarios de taux d'enregistrement ; pas de taux de capture estimé. | [`137`](2_donnees/137_descente_diagnostics.py), [`138`](2_donnees/138_descente_observation.py) ; [sortie 137](../../resultats/sorties_verif_inst/137.txt), [sortie 138](../../resultats/sorties_verif_inst/138.txt). | Les mêmes comptes observés peuvent correspondre à plusieurs fréquences réelles ; le bootstrap ne résout pas cette non-identification. |
| **Validation annuelle — diagnostic** | Comparaison hors échantillon de la loi annuelle et des observations. | [`91`](1_fondations/91_backtest_charge_agregee.py) ; [sortie Institut 91](../../resultats/sorties_verif_inst/91.txt). | Ajustement du niveau rejeté, p = 0,0042 ; les contrôles techniques ne constituent pas une validation du niveau de capital. |

## Expériences historiques à reconnaître

Le [diagnostic de repli logit](6_contagion/34_echelle_repli_u.py) est distinct du pavé à S fixé :
il compare des étendues sur 400 tirages par famille. Sa famille source/cible n'impose pas un
rang matriciel 1 après recentrage ; `u = 0` conserve les attractivités des cibles et une asymétrie.
S varie aussi : ce diagnostic n'isole pas l'effet de la partie antisymétrique.

| Entrée | Statut et usage |
|---|---|
| [`60_descente_echelle_entite.py`](2_donnees/60_descente_echelle_entite.py), ancienne valeur 169 M€ | **Historique**. Comparaison avec la nouvelle descente ; ne pas l'utiliser comme estimation courante. |
| [`src/aggregation/lda.py`](../../src/aggregation/lda.py), chaîne à plusieurs briques | **Chaîne antérieure / fonctions réutilisées**. Le moteur courant importe notamment sa fonction de sévérité ; il n'active pas pour autant toutes ses anciennes briques. |
| Scripts `16`, `16b` et `18` | **Lectures partielles / références de comparaison**. Les canaux et les priorités ne sont pas ceux du calcul à quatre canaux de `140` : lecture B, P1 puis P4 ; quatre canaux, P4 puis P1, sous les barèmes posés. |
| [`exemples/calcul_synthetique.py`](exemples/calcul_synthetique.py) | **Pédagogique**. Paramètres fictifs, cascade indépendante ; démontre le chemin du calcul, sans recalibrer le mémoire. |

Les identifiants des scripts sont conservés pour préserver les citations et les sorties. Leur
numéro n'indique ni leur priorité ni leur statut scientifique. Les sorties dans
`resultats/sorties_verif_inst/` remplacent les sorties de même nom dans `sorties_verif/` pour
le manuscrit Institut ; elles ne réécrivent pas les versions historiques.

## Ce que les contrôles établissent

- **Tests scientifiques** : propriétés du code sur des cas contrôlés, souvent synthétiques.
- **Harnais** : correspondance des nombres du manuscrit avec les sorties enregistrées.
- **Compilation** : cohérence des références, de la mise en page et de la pagination.
- **Validation empirique** : confrontation à des observations indépendantes ; elle reste partielle.

Le dépôt permet les trois premiers contrôles sans bases brutes. Il ne transforme pas les
paramètres de scénario en paramètres estimés. Pour commencer par un calcul entièrement
accessible, suivre l'[exemple synthétique](exemples/README.md).
