"""Prépare un second codage SAS en aveugle et mesure l'accord après réception réelle.

Les textes sous licence et la clé du premier codage restent dans le dossier local ignoré par Git.
Le questionnaire n'affiche ni les identifiants SAS ni les codes du premier lecteur.
"""
import argparse
import hashlib
import html
import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "recherche/vasicek_lab"
LOCAL = ROOT / "docs/collaboration/local/validation_codage_sas"
GRILLE = {
    1: "Gestion des risques TIC : accès, authentification, données, conception, changements, continuité et secours.",
    2: "Gestion des incidents : détection tardive, réponse ou notification défaillante.",
    3: "Tests : tests insuffisants, vulnérabilité connue non corrigée, absence de tests d'intrusion.",
    4: "Prestataires : défaillance d'un fournisseur, éditeur, sous-traitant ou infrastructure contribuant à l'incident.",
    5: "Partage d'information : alerte reçue non exploitée, ou information de menace non transmise.",
}


def charger_codes():
    codes = []
    for name in ("pilote_codage_sas", "codage_sas_complet"):
        spec = importlib.util.spec_from_file_location(name, LAB / "6_contagion" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        codes.extend(module.CODES)
    if len(codes) != 570 or len({str(c[0]) for c in codes}) != 570:
        raise ValueError("Le premier codage doit contenir 570 identifiants uniques.")
    return codes


def accord_binaire(a, b):
    if len(a) != len(b) or not a:
        raise ValueError("Deux séries non vides de même taille sont nécessaires.")
    n = len(a)
    n11 = sum(x and y for x, y in zip(a, b))
    n00 = sum(not x and not y for x, y in zip(a, b))
    n10 = sum(x and not y for x, y in zip(a, b))
    n01 = n - n11 - n00 - n10
    po = (n11 + n00) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return dict(n=n, n11=n11, n00=n00, n10=n10, n01=n01, accord=po,
                kappa=(po - pe) / (1 - pe) if pe < 1 else None)


def evaluer(reponses, cle):
    if not str(reponses.get("codeur", "")).strip() or not reponses.get("aveugle_declare"):
        raise ValueError("L'identité du codeur et sa déclaration de lecture en aveugle sont nécessaires.")
    attendus = {c["cas"]: c for c in cle["cas"]}
    rendus = reponses.get("cas", [])
    ids = [c.get("cas") for c in rendus]
    if len(ids) != len(set(ids)) or set(ids) != set(attendus):
        raise ValueError("Cas manquants, inconnus ou dupliqués.")
    for c in rendus:
        if c.get("termine") is not True:
            raise ValueError(f"{c['cas']} : codage non terminé ; une absence de réponse n'est pas un code vide.")
        for niveau in ("strict", "large"):
            vals = c.get(niveau)
            if (not isinstance(vals, list) or len(vals) != len(set(vals))
                    or any(type(v) is not int or v not in range(1, 6) for v in vals)):
                raise ValueError(f"{c['cas']} : codes {niveau} invalides.")
        if not set(c["strict"]) <= set(c["large"]):
            raise ValueError(f"{c['cas']} : le codage strict doit être inclus dans le large.")
        if not str(c.get("justification", "")).strip():
            raise ValueError(f"{c['cas']} : justification absente.")
    mesures = {}
    for niveau in ("strict", "large"):
        mesures[niveau] = {}
        for strate in ("ensemble", "E", "B"):
            sous = [c for c in rendus if strate == "ensemble" or attendus[c["cas"]]["strate"] == strate]
            if not sous:
                continue
            mesures[niveau][strate] = {
                f"P{p}": accord_binaire(
                    [p in attendus[c["cas"]][niveau] for c in sous], [p in c[niveau] for c in sous])
                for p in range(1, 6)
            }
            mesures[niveau][strate]["accord_ensemble_exact"] = sum(
                set(c[niveau]) == set(attendus[c["cas"]][niveau]) for c in sous) / len(sous)
    return dict(codeur=reponses["codeur"], n=len(rendus), mesures=mesures,
                independance="Déclarée par le codeur, non vérifiable automatiquement.",
                portee="Accord descriptif dans un échantillon stratifié ; ni vérité des codes ni estimation de S ou A.")


def preparer(destination, taille, graine):
    destination = destination.resolve()
    if not destination.is_relative_to(LOCAL.resolve()):
        raise ValueError("Les textes doivent rester dans docs/collaboration/local/validation_codage_sas/.")
    if taille < 2 or taille % 2 or taille // 2 > 99:
        raise ValueError("Choisir une taille paire entre 2 et 198 (moitié par strate).")
    destination.mkdir(parents=True, exist_ok=True)
    if any((destination / f).exists() for f in ("cle.json", "questionnaire.html", "reponses_vierges.json")):
        raise ValueError("Ce lot existe déjà ; choisir un nouveau dossier pour conserver les réponses.")
    codes = charger_codes()
    rng = random.Random(graine)
    strates = {"E": [], "B": []}
    for ident, cat, strict, large in codes:
        strate = "E" if cat in ("E", "EF") else "B"
        strates[strate].append((str(ident), strate, sorted(strict), sorted(large)))
    tires = [c for strate in ("E", "B") for c in rng.sample(strates[strate], taille // 2)]
    rng.shuffle(tires)
    sys.path.insert(0, str(ROOT))
    from src.severity.oprisk_analysis import load_clean, filter_cyber, filter_finance
    base = filter_finance(filter_cyber(load_clean(ROOT / "data/raw/SAS_OpRisk_Global_Data_June_2026.xlsx")))
    base = base[base["year"].between(2000, 2026)]
    def identifiant(v):
        return str(int(v)) if str(v).replace(".0", "").isdigit() else str(v).strip()
    descriptions = {identifiant(r["Reference ID Code"]): str(r["Description of Event"])
                    for _, r in base.iterrows()}
    reponses, secrets, cartes = [], [], []
    for i, (ident, strate, strict, large) in enumerate(tires, 1):
        cas = f"V{i:03d}"
        description = descriptions.get(ident)
        if not description or description == "nan":
            raise ValueError(f"Description manquante pour {cas}.")
        reponses.append(dict(cas=cas, termine=False, strict=None, large=None, justification=""))
        secrets.append(dict(cas=cas, identifiant=ident, strate=strate, strict=strict, large=large))
        groupes = "".join(
            f'<fieldset><legend>{niveau}</legend>' + "".join(
                f'<label><input type="checkbox" data-niveau="{niveau}" value="{p}">P{p}</label> '
                for p in range(1, 6)) + '</fieldset>' for niveau in ("strict", "large"))
        cartes.append(f'<article data-cas="{cas}"><h2>{cas}</h2><p class="description">{html.escape(description)}</p>'
                      + groupes + '<label>Justification (preuve ou absence d’information)<textarea></textarea></label>'
                      '<label><input type="checkbox" data-termine>Cas terminé, même si aucun pilier n’est retenu</label></article>')
    paquet = dict(codeur="", aveugle_declare=False, cas=reponses)
    cle = dict(graine=graine, taille=taille, population={k: len(v) for k, v in strates.items()}, cas=secrets)
    (destination / "cle.json").write_text(json.dumps(cle, ensure_ascii=False, indent=2), encoding="utf8")
    (destination / "reponses_vierges.json").write_text(json.dumps(paquet, ensure_ascii=False, indent=2), encoding="utf8")
    grille = "".join(f'<li><b>P{p}</b> {html.escape(s)}</li>' for p, s in GRILLE.items())
    page = '''<!doctype html><html lang="fr"><meta charset="utf-8"><title>Second codage SAS</title>
<style>body{max-width:900px;margin:40px auto;font:17px/1.5 system-ui;color:#243440}article{border-top:2px solid #a52d39;padding:20px 0;margin:30px 0}textarea{display:block;width:95%;min-height:70px}fieldset{margin:12px 0}fieldset label{margin-right:20px}.description{white-space:pre-wrap}header{background:#f2f6f7;padding:20px}button{padding:12px;margin:20px 0}</style>
<header><h1>Second codage des descriptions SAS</h1><p>Travail local sur des textes sous licence. Seul le questionnaire est destiné au second codeur autorisé ; conserver la clé séparément. Ne pas consulter les codes précédents.</p>
<p>Strict : défaillance établie par le texte. Large : strict, plus défaillance impliquée sans être établie. Une intrusion ou une fraude ne prouve pas à elle seule une défaillance de contrôle. Aucun sens de transmission n’est demandé.</p><ul>GRILLE</ul>
<p>Échantillon équilibré entre fraudes externes et interruptions : il ne représente pas leurs proportions dans la base. Aucun résultat du premier codage n’est affiché.</p>
<label>Codeur <input id="codeur"></label><p><label><input type="checkbox" id="aveugle">Je déclare avoir codé sans consulter le premier codage.</label></p></header>
CARTES
<button id="exporter">Télécharger les réponses JSON</button><p id="etat"></p>
<script>document.getElementById('exporter').onclick=()=>{
const cas=[...document.querySelectorAll('article')].map(a=>({cas:a.dataset.cas,termine:a.querySelector('[data-termine]').checked,
strict:[...a.querySelectorAll('[data-niveau="strict"]:checked')].map(x=>+x.value),large:[...a.querySelectorAll('[data-niveau="large"]:checked')].map(x=>+x.value),justification:a.querySelector('textarea').value}));
const obj={codeur:document.getElementById('codeur').value,aveugle_declare:document.getElementById('aveugle').checked,cas};
const url=URL.createObjectURL(new Blob([JSON.stringify(obj,null,2)],{type:'application/json'}));const lien=document.createElement('a');lien.href=url;lien.download='reponses_second_codeur.json';lien.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
document.getElementById('etat').textContent=cas.filter(c=>c.termine).length+' cas terminés sur '+cas.length+'. Le fichier téléchargé conserve les réponses ; aucune donnée n’est envoyée.';};</script></html>'''
    (destination / "questionnaire.html").write_text(page.replace("GRILLE", grille).replace("CARTES", "".join(cartes)), encoding="utf8")
    print(f"Lot préparé : {taille} cas, {taille // 2} par strate. Dossier : {destination}")
    print("Aucun accord calculé : les réponses d'un second codeur sont encore attendues.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commandes = parser.add_subparsers(dest="action", required=True)
    prep = commandes.add_parser("preparer")
    prep.add_argument("--dossier", type=Path, default=LOCAL / "lot_100_20261027")
    prep.add_argument("--taille", type=int, default=100)
    prep.add_argument("--graine", type=int, default=20261027)
    accord = commandes.add_parser("evaluer")
    accord.add_argument("--reponses", type=Path, required=True)
    accord.add_argument("--cle", type=Path, required=True)
    accord.add_argument("--sortie", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "preparer":
        preparer(args.dossier, args.taille, args.graine)
    else:
        rapport = evaluer(json.loads(args.reponses.read_text(encoding="utf-8-sig")),
                          json.loads(args.cle.read_text(encoding="utf-8-sig")))
        args.sortie.write_text(json.dumps(rapport, ensure_ascii=False, indent=2), encoding="utf8")
        print(f"Accord descriptif calculé pour {rapport['n']} cas. Rapport : {args.sortie}")


if __name__ == "__main__":
    main()
