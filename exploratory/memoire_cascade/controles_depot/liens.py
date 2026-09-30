# -*- coding: utf-8 -*-
"""Statut HTTP de chaque lien externe du PDF depose.

Usage, depuis exploratory/memoire_cascade/ :
    ..\\..\\.venv\\Scripts\\python.exe controles_depot\\liens.py

Au 29 septembre 2026, les 22 liens repondent (200 ou 202). Deux sites, la BCE et Marsh, ont pu
faire echouer la verification de certificat du Python du poste sans etre morts : en cas d'erreur
SSL, les recontroler par Invoke-WebRequest, qui passe par le magasin de certificats de Windows.
"""
import os
import urllib.error
import urllib.request

import fitz

MEMOIRE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = fitz.open(os.path.join(MEMOIRE, "main_ensae.pdf"))
urls = sorted({l["uri"] for p in d for l in p.get_links() if l.get("uri")})
ENTETE = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/128.0 Safari/537.36",
          "Accept": "text/html,application/pdf,*/*"}
echecs = 0
for u in urls:
    try:
        with urllib.request.urlopen(urllib.request.Request(u, headers=ENTETE), timeout=25) as r:
            print(f"{r.status}  {u}" + (f"\n     -> {r.url}" if r.url != u else ""))
    except urllib.error.HTTPError as e:
        echecs += 1
        print(f"{e.code}  {u}")
    except Exception as e:
        echecs += 1
        print(f"ERR  {u}  ({type(e).__name__}: {str(e)[:70]})")
print(f"\n{len(urls)} liens, {echecs} a recontroler")
