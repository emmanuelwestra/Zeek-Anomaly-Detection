from __future__ import annotations

import numpy as np
from conftest import synthetic_flows

from ndr.features import FlowPreprocessor
from ndr.models import OCSVM, MixedAutoencoder


def test_feature_contract_excludes_labels_and_maps_unknown_categories() -> None:
    train = synthetic_flows(20)
    test = train.iloc[:2].copy()
    test["service"] = "unseen-service"
    oc = FlowPreprocessor.for_model("ocsvm", {"top_k_dest_ports": 4})
    ae = FlowPreprocessor.for_model("ae", {"top_k_dest_ports": 4})
    x_oc = oc.fit_transform(train)
    x_ae = ae.fit_transform(train)
    transformed = ae.transform(test)
    assert x_ae.shape[1] == x_oc.shape[1] + 11
    assert transformed.shape == (2, x_ae.shape[1])
    assert all("uid" not in name and "attack" not in name for name in ae.feature_names_)


def test_static_models_score_anomalies_and_round_trip(tmp_path) -> None:
    frame = synthetic_flows(24)
    oc_pre = FlowPreprocessor.for_model("ocsvm", {"top_k_dest_ports": 4})
    ae_pre = FlowPreprocessor.for_model("ae", {"top_k_dest_ports": 4})
    x_oc, x_ae = oc_pre.fit_transform(frame), ae_pre.fit_transform(frame)
    oc = OCSVM(nu=0.1, scoring_threads=1).fit(x_oc)
    ae = MixedAutoencoder(
        ae_pre.reconstruction_schema(), hidden_dims=[8], latent_dim=2, epochs=2,
        batch_size=8, patience=1, internal_validation_fraction=0.1, device="cpu",
    ).fit(x_ae)
    assert np.isfinite(oc.score_samples(x_oc)).all()
    assert np.isfinite(ae.score_samples(x_ae)).all()
    oc.save(tmp_path / "oc.joblib")
    ae.save(tmp_path / "ae.pt")
    np.testing.assert_allclose(OCSVM.load(tmp_path / "oc.joblib").score_samples(x_oc), oc.score_samples(x_oc))
    np.testing.assert_allclose(MixedAutoencoder.load(tmp_path / "ae.pt").score_samples(x_ae), ae.score_samples(x_ae))
