# Une année d'incidents jusqu'au capital

Cet exemple utilise des paramètres **entièrement fictifs**. Il montre les quatre canaux et
le chemin du calcul sans base privée. Il réutilise `card_dist_all` du moteur de cascade
indépendante ; il ne reproduit pas le résultat principal du mémoire, obtenu par une marche
auto-évitante. Les [protocoles du mémoire](../CARTE_CALCULS.md) sont distincts.

Depuis la racine, avec les dépendances de `requirements.txt` installées :

```bash
python recherche/vasicek_lab/exemples/calcul_synthetique.py
```

La commande n'écrit aucun fichier par défaut. Pour conserver les paramètres, les mesures
et les incidents détaillés dans un JSON :

```bash
python recherche/vasicek_lab/exemples/calcul_synthetique.py --annees 50000 --graine 42 --sortie build/exemple_synthetique.json
```

## Les conventions de l'exemple

| Entrée | Conforme fictif | Non conforme fictif | Statut |
|---|---:|---:|---|
| Fréquence annuelle λ | 2 | 3,5 | Posée pour l'exemple |
| Probabilité d'une perte matérielle p_u, par pilier touché | 0,10 | 0,16 | Posée |
| Gain g appliqué à la matrice de liens | 0,30 | 0,60 | Posé |
| Inclusion de chacun des autres piliers lors d'une amorce P4, ω | 0 | 0,45 | Posée |

La fréquence suit une binomiale négative, avec variance égale à deux fois la moyenne.
Les amorces sont équiprobables entre les cinq piliers. La matrice de liens, écrite dans le
script et dans le JSON, est fictive. Lors d'une amorce P4, le mécanisme de choc commun
**remplace** la cascade : P4 et chacun des autres piliers avec probabilité ω. À ω = 0,
une amorce P4 ne touche que P4. Cette convention est propre à l'exemple : dans le moteur
publié, l'état conforme n'a pas de choc commun et P4 y reste un nœud de cascade.

Les piliers touchés tirent chacun une perte matérielle avec probabilité p_u. Sinon leur
perte vaut zéro : cet exemple ne modélise pas les petites pertes sous le seuil. Au-delà
du seuil fictif de 1 M€, l'excès suit une Pareto généralisée de forme ξ = 0,30 et d'échelle
σ = 0,50 M€. Les pertes élémentaires sont indépendantes conditionnellement aux piliers atteints.
Cette loi sans plafond a une moyenne et une variance finies ; l'ES est finie.

## Lire le calcul affiché

1. On tire le nombre d'incidents de chaque année.
2. Chaque incident commence sur un pilier ; la cascade ou le choc commun détermine l'ensemble touché.
3. On additionne les pertes des piliers touchés pour obtenir celle de l'incident.
4. On additionne les incidents pour obtenir la perte annuelle L.
5. Le quantile empirique à 99,5 % des années donne la VaR. L'ES moyenne la fraction de queue
   de 0,5 %, avec traitement des égalités au quantile.
6. On compare les VaR des deux états : cet écart diffère de la variation des pertes moyennes.

Le programme affiche une année avec perte positive pour rendre la somme vérifiable. Cette
année choisie pour l'explication n'est pas présentée comme représentative. Il calcule aussi
la moyenne théorique à partir de λ, du nombre moyen de piliers touchés et de la perte moyenne
par pilier : cette quantité permet de contrôler la simulation indépendamment du quantile.

Les états partagent leurs uniformes par année, rang d'incident et pilier. Les incidents
supplémentaires d'un état occupent des rangs supplémentaires. Cet appariement réduit certains
écarts de simulation ; il ne supprime pas le bruit et ne constitue pas un effet causal.
Avec 20 000 années, la queue à 0,5 % ne contient que 100 années : les chiffres affichés
illustrent le calcul, sans intervalle de précision ni calibration d'un capital réel.

Les tests vérifient la réconciliation des incidents et de l'année, la moyenne analytique,
le cas sans perte matérielle, l'appariement de deux états identiques et le calcul d'ES avec
des égalités. Ils sont exécutés avec les autres tests scientifiques :

```bash
python -m pytest recherche/vasicek_lab/tests/test_exemple_synthetique.py -q
```
