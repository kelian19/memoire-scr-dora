"""Contrats des fonctions de production, cas analytiques et références à graines fixes."""
from pathlib import Path
import sys

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1]
for chemin in (LAB.parents[1], LAB):
    sys.path.insert(0, str(chemin))
import canaux_conformite as cx
import partial_id as pid


def pertes(**changements):
    args = dict(lam=2., g=.45, p_u=.15, phi_cs=None,
                rng=np.random.default_rng(121), ny=25)
    args.update(changements)
    return cx.pertes_annuelles(**args)


def test_frequence_nulle_ne_consomme_pas_de_tirages():
    rng = np.random.default_rng(121)
    attendu = np.random.default_rng(121).random(5)
    np.testing.assert_array_equal(pertes(lam=0., rng=rng), np.zeros(25))
    np.testing.assert_array_equal(rng.random(5), attendu)


@pytest.mark.parametrize("nom,valeur", [
    ("lam", -1.), ("lam", np.nan), ("lam", np.inf), ("lam", [2.]),
    ("g", -0.1), ("g", 1.5), ("g", np.nan), ("g", .5+.1j),
    ("p_u", -0.1), ("p_u", 1.5), ("p_u", np.inf),
    ("phi_cs", -0.1), ("phi_cs", 1.5), ("phi_cs", np.nan),
    ("ny", 0), ("ny", -1), ("ny", 2.5), ("ny", True),
])
def test_parametres_invalides_refuses_avant_simulation(nom, valeur):
    with pytest.raises(ValueError, match=nom):
        pertes(**{nom: valeur})


def test_frequence_nulle_ne_masque_pas_un_parametre_invalide():
    with pytest.raises(ValueError, match="p_u"):
        pertes(lam=0., p_u=1.5)


def test_sans_propagation_perte_constante_par_incident(monkeypatch):
    # Chaque incident touche un seul pilier et coûte 4 : L_année = 4 N_année.
    # Cela vérifie l'affectation aux années, avec des comptes et coûts connus.
    monkeypatch.setattr(cx, "simulate_remediation_severity",
                        lambda n, *args: np.full(n, 4.))
    lam, ny, seed = 3., 100, 121
    r = lam / (cx.ec.PHI - 1.)
    comptes = np.random.default_rng(seed).negative_binomial(r, r/(r+lam), size=ny)
    observe = pertes(lam=lam, g=0., p_u=1., ny=ny, rng=np.random.default_rng(seed))
    np.testing.assert_array_equal(observe, 4*comptes)


def test_p4_ordinaire_et_choc_nul_sont_distincts():
    indicatrices, probas = cx.table_p4_choc(0.)
    assert probas.sum() == pytest.approx(1.)
    assert probas @ indicatrices.sum(axis=1) == pytest.approx(1.)
    normales, poids = cx.table_amorce(4, .9)
    assert poids @ normales.sum(axis=1) > 1.
    certaines, poids = cx.table_p4_choc(1.)
    assert poids @ certaines.sum(axis=1) == pytest.approx(5.)


def test_tables_refusent_les_probabilites_invalides():
    with pytest.raises(ValueError, match="phi_cs"):
        cx.table_p4_choc(1.01)
    with pytest.raises(ValueError, match="g"):
        cx.table_amorce(1, np.nan)
    with pytest.raises(ValueError, match="nseed"):
        cx.metriques_par_graine(2., .45, .15, nseed=0)


def test_admissibilite_protege_le_domaine_sans_restreindre_les_sommets():
    zero = np.zeros((5, 5))
    assert pid.admissible(zero)
    for w in (np.eye(5)*.1, np.zeros((4, 4)), np.full((5, 5), np.nan), zero+1j):
        assert not pid.admissible(w)
    unite = zero.copy()
    unite[0, 1] = unite[1, 0] = 1.
    assert not pid.admissible(unite)  # rho = 1, et non strictement inférieur.
    s, _ = pid.decompose(pid.expert_matrix())
    matrices = [pid.build_W(s, a) for a in pid.all_vertices(s, 1.)]
    assert all(pid.admissible(w) for w in matrices)
    assert any(w.sum(axis=1).max() > 1 for w in matrices)


def test_references_quatre_canaux_sont_conservees():
    # Référence de l'audit 2923d34 : protège aussi l'ordre et le nombre de tirages.
    c = cx.scr_par_graine(cx.LAM_C, cx.G_C, cx.PU_C, None)
    nc = cx.scr_par_graine(cx.LAM_NC, cx.G_NC, cx.PU_NC, cx.PHICS_NC)
    assert c.mean() == pytest.approx(6048.959395, abs=1e-6, rel=0)
    assert nc.mean() == pytest.approx(20187.675096, abs=1e-6, rel=0)
    assert (nc-c).std(ddof=1) == pytest.approx(733.955829, abs=1e-6, rel=0)
