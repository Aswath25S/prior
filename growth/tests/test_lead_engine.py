import unittest

import pandas as pd

from growth.lead_engine import (
    build_account_view,
    build_lead_view,
    next_lead_id,
    persist_owner_assignments,
    record_calendly_invites,
    score_lead,
    update_lead_progress,
    waiting_points,
)


class AssignmentStabilityTest(unittest.TestCase):
    def setUp(self):
        self.team = pd.DataFrame([
            {
                "salesperson_id": "S001",
                "name": "Sales One",
                "weekly_capacity": 1,
                "calendly_url": "https://example.com/sales-one",
            },
        ])
        self.leads = pd.DataFrame([
            self._lead("L001", 500, "S001", "A defined demand forecasting project"),
            self._lead("L002", 5000, "UNASSIGNED", "Production deployment and procurement"),
        ])

    @staticmethod
    def _lead(lead_id, company_size, owner_id, inbound_text):
        return {
            "lead_id": lead_id,
            "name": lead_id,
            "email": f"contact@{lead_id.lower()}.example.com",
            "company": f"Company {lead_id}",
            "company_size": company_size,
            "job_title": "Director of Data Science",
            "source": "Direct website",
            "inbound_time": "2026-09-27T12:00:00Z",
            "account_activity_count": 1,
            "inbound_text": inbound_text,
            "owner_id": owner_id,
            "status": "In Progress",
        }

    def test_highest_scoring_lead_receives_limited_capacity(self):
        view = build_lead_view(self.leads, self.team).set_index("lead_id")

        self.assertEqual(view.at["L001", "owner_id"], "UNASSIGNED")
        self.assertEqual(view.at["L002", "owner_id"], "S001")

    def test_higher_scoring_new_lead_takes_limited_capacity(self):
        self.leads.loc[self.leads["lead_id"] == "L001", "status"] = "Success"
        view = build_lead_view(self.leads, self.team)

        self.assertEqual(persist_owner_assignments(self.leads, view), 1)
        self.assertEqual(
            self.leads.loc[self.leads["lead_id"] == "L002", "owner_id"].iloc[0],
            "S001",
        )

        higher_scoring_lead = self._lead(
            "L003", 50000, "UNASSIGNED", "Production security procurement"
        )
        higher_scoring_lead["source"] = "SAP"
        self.leads = pd.concat([
            self.leads,
            pd.DataFrame([higher_scoring_lead]),
        ], ignore_index=True)
        refreshed = build_lead_view(self.leads, self.team).set_index("lead_id")

        self.assertEqual(refreshed.at["L002", "owner_id"], "UNASSIGNED")
        self.assertEqual(refreshed.at["L003", "owner_id"], "S001")

    def test_success_records_learning_fields_and_score_snapshot(self):
        view = build_lead_view(self.leads, self.team)
        snapshot = view[view["lead_id"] == "L001"].iloc[0]

        update_lead_progress(
            self.leads,
            snapshot,
            "Success",
            success_milestone="Commercial win",
            opportunity_created=True,
            opportunity_stage="Closed won",
            estimated_value_band="$50k–$200k",
            next_action="Schedule technical discovery",
            next_action_due="2026-10-01",
            now="2026-09-27T18:00:00Z",
        )

        recorded = self.leads.set_index("lead_id").loc["L001"]
        self.assertEqual(recorded["qualification_outcome"], "Qualified")
        self.assertEqual(recorded["success_milestone"], "Commercial win")
        self.assertTrue(recorded["opportunity_created"])
        self.assertEqual(recorded["opportunity_stage"], "Closed won")
        self.assertEqual(recorded["qualification_score_at_outcome"], snapshot["qualification_score"])
        self.assertEqual(recorded["outcome_recorded_at"], "2026-09-27T18:00:00+00:00")

    def test_rejection_requires_a_disposition_reason(self):
        view = build_lead_view(self.leads, self.team)
        snapshot = view[view["lead_id"] == "L001"].iloc[0]

        with self.assertRaisesRegex(ValueError, "disposition reason"):
            update_lead_progress(self.leads, snapshot, "Rejected")

    def test_qualified_and_poc_stages_remain_in_progress(self):
        view = build_lead_view(self.leads, self.team)
        snapshot = view[view["lead_id"] == "L001"].iloc[0]

        update_lead_progress(
            self.leads,
            snapshot,
            "In Progress",
            pipeline_stage="Security review and POC design",
            estimated_value_band="$50k–$200k",
        )

        recorded = self.leads.set_index("lead_id").loc["L001"]
        self.assertEqual(recorded["status"], "In Progress")
        self.assertEqual(recorded["pipeline_stage"], "Security review and POC design")
        self.assertTrue(recorded["opportunity_created"])
        self.assertEqual(recorded["opportunity_stage"], "POC")

    def test_closing_bundled_account_releases_one_slot_for_next_account(self):
        sibling = self._lead("L003", 500, "S001", "Second contact at the same account")
        sibling["email"] = self.leads.iloc[0]["email"]
        next_lead = self._lead(
            "L004", 50, "UNASSIGNED", "Exploring whether this might be useful"
        )
        leads = pd.concat(
            [self.leads.iloc[[0]], pd.DataFrame([sibling, next_lead])],
            ignore_index=True,
        )
        initial_view = build_lead_view(leads, self.team)
        bundled_id = initial_view.loc[initial_view["lead_id"] == "L001", "account_id"].iloc[0]
        account = build_account_view(initial_view)
        account = account[account["account_id"] == bundled_id].iloc[0]

        update_lead_progress(leads, account, "Success")
        refreshed = build_lead_view(leads, self.team)

        self.assertEqual(
            set(refreshed.loc[refreshed["account_id"] == bundled_id, "status"]),
            {"Success"},
        )
        self.assertEqual(
            refreshed.loc[refreshed["lead_id"] == "L004", "owner_id"].iloc[0],
            "S001",
        )
        active_for_rep = build_account_view(
            refreshed[
                (refreshed["owner_id"] == "S001")
                & (refreshed["status"] == "In Progress")
            ]
        )
        self.assertEqual(len(active_for_rep), 1)

    def test_contacts_at_same_account_share_owner_and_one_capacity_slot(self):
        second_contact = self._lead(
            "L003", 500, "UNASSIGNED", "Security review for the same evaluation"
        )
        second_contact["email"] = self.leads.iloc[0]["email"]
        leads = pd.concat([self.leads.iloc[[0]], pd.DataFrame([second_contact])], ignore_index=True)

        view = build_lead_view(leads, self.team)

        self.assertEqual(view["account_id"].nunique(), 1)
        self.assertEqual(set(view["owner_id"]), {"S001"})
        self.assertEqual(set(view["account_contact_count"]), {2})

    def test_independent_and_not_provided_leads_are_never_bundled(self):
        leads = pd.DataFrame([
            self._lead("L010", 0, "UNASSIGNED", "Independent inquiry"),
            self._lead("L011", 0, "UNASSIGNED", "Another independent inquiry"),
            self._lead("L012", 0, "UNASSIGNED", "Anonymous inquiry"),
            self._lead("L013", 0, "UNASSIGNED", "Another anonymous inquiry"),
        ])
        leads.loc[leads["lead_id"].isin(["L010", "L011"]), "company"] = "Independent"
        leads.loc[leads["lead_id"].isin(["L012", "L013"]), "company"] = "Not provided"
        leads["email"] = "shared@gmail.com"
        leads["account_id"] = "incoming-shared-account"

        view = build_lead_view(leads, self.team)

        self.assertEqual(view["account_id"].nunique(), 4)
        self.assertEqual(set(view["account_contact_count"]), {1})

    def test_messy_company_size_does_not_crash_scoring(self):
        for value in [None, float("nan"), "unknown"]:
            result = score_lead({
                "company_size": value,
                "job_title": None,
                "inbound_text": None,
                "source": "Other",
            })
            self.assertIsInstance(result["score"], int)

        self.assertEqual(score_lead({
            "company_size": "25,000",
            "job_title": "",
            "inbound_text": "",
            "source": "Other",
        })["account_type"], "Large enterprise")

    def test_missing_lead_fields_receive_safe_fallbacks(self):
        leads = pd.DataFrame([{"inbound_time": "not-a-date"}])

        view = build_lead_view(leads, self.team)
        lead = view.iloc[0]

        self.assertEqual(lead["lead_id"], "AUTO-001")
        self.assertEqual(lead["name"], "Unknown contact")
        self.assertEqual(lead["company"], "Not provided")
        self.assertEqual(lead["email"], "Not provided")
        self.assertEqual(lead["job_title"], "Unknown role")
        self.assertEqual(lead["inbound_text"], "No inbound message provided.")
        self.assertEqual(lead["confidence"], "Low")
        self.assertIn("email", lead["missing_data_fields"])
        self.assertTrue(pd.isna(lead["received_at"]))
        self.assertEqual(record_calendly_invites(leads, view, self.team), [])

    def test_next_lead_id_ignores_nonstandard_fallback_ids(self):
        leads = pd.DataFrame({"lead_id": ["L009", "AUTO-001", None, "external-42"]})

        self.assertEqual(next_lead_id(leads), "L010")

    def test_waiting_time_changes_priority_but_not_commercial_quality(self):
        early = self._lead("L004", 500, "UNASSIGNED", "Demand forecasting evaluation")
        late = self._lead("L005", 500, "UNASSIGNED", "Demand forecasting evaluation")
        now = pd.Timestamp.now(tz="UTC")
        early["inbound_time"] = (now - pd.Timedelta(days=3)).isoformat()
        late["inbound_time"] = (now - pd.Timedelta(hours=1)).isoformat()

        view = build_lead_view(pd.DataFrame([early, late]), self.team).set_index("lead_id")

        self.assertEqual(
            view.at["L004", "qualification_score"], view.at["L005", "qualification_score"]
        )
        self.assertGreater(view.at["L004", "score"], view.at["L005", "score"])

    def test_waiting_time_uses_recorded_timezone_aware_inbound_time(self):
        lead = self._lead("L020", 500, "UNASSIGNED", "Demand forecasting evaluation")
        lead["inbound_time"] = "2026-09-27T12:00:00-04:00"

        view = build_lead_view(
            pd.DataFrame([lead]),
            self.team,
            now="2026-09-27T18:30:00-04:00",
        ).iloc[0]

        self.assertAlmostEqual(view["waiting_hours"], 6.5)
        self.assertAlmostEqual(view["waiting_days"], 6.5 / 24)
        self.assertEqual(view["waiting"], "6h 30m")
        self.assertEqual(view["time_points"], 0)

    def test_waiting_score_changes_at_day_boundaries(self):
        expected_points = {
            0.99: 0,
            1: 2,
            2: 4,
            3: 7,
            5: 11,
            7: 15,
        }

        for days, points in expected_points.items():
            with self.subTest(days=days):
                self.assertEqual(waiting_points(days), points)

    def test_self_serve_requires_low_quality_and_seven_days_waiting(self):
        now = pd.Timestamp("2026-09-28T12:00:00Z")
        old_low_quality = self._lead(
            "L030", 1, "UNASSIGNED", "Personal project and learning only"
        )
        recent_low_quality = self._lead(
            "L031", 1, "UNASSIGNED", "Personal project and learning only"
        )
        old_high_quality = self._lead(
            "L032", 5000, "UNASSIGNED", "Production deployment and procurement"
        )
        old_low_quality["inbound_time"] = (now - pd.Timedelta(days=7)).isoformat()
        recent_low_quality["inbound_time"] = (
            now - pd.Timedelta(days=6, hours=23)
        ).isoformat()
        old_high_quality["inbound_time"] = (now - pd.Timedelta(days=8)).isoformat()

        view = build_lead_view(
            pd.DataFrame([old_low_quality, recent_low_quality, old_high_quality]),
            self.team,
            now=now,
        ).set_index("lead_id")

        self.assertLess(view.at["L030", "qualification_score"], 30)
        self.assertTrue(view.at["L030", "self_serve"])
        self.assertEqual(view.at["L030", "engagement"], "Self-serve")
        self.assertFalse(view.at["L031", "self_serve"])
        self.assertGreaterEqual(view.at["L032", "qualification_score"], 30)
        self.assertFalse(view.at["L032", "self_serve"])

    def test_closed_lead_waiting_time_stops_at_outcome(self):
        lead = self._lead("L021", 500, "S001", "Demand forecasting evaluation")
        lead["inbound_time"] = "2026-09-27T12:00:00Z"
        lead["status"] = "Success"
        lead["outcome_recorded_at"] = "2026-09-27T17:00:00Z"

        view = build_lead_view(
            pd.DataFrame([lead]),
            self.team,
            now="2026-09-28T12:00:00Z",
        ).iloc[0]

        self.assertAlmostEqual(view["waiting_hours"], 5.0)
        self.assertEqual(view["waiting"], "5h 0m")

    def test_explicit_evaluation_outranks_large_account_without_active_project(self):
        inactive = score_lead({
            "company_size": 78000,
            "job_title": "Innovation Director",
            "source": "SAP",
            "inbound_text": "Interesting technology, but no active project right now.",
            "account_activity_count": 1,
            "account_contact_count": 1,
        })
        active = score_lead({
            "company_size": 12000,
            "job_title": "ML Platform Lead",
            "source": "Free API",
            "inbound_text": "Several teams are evaluating private networking, security, and pricing.",
            "account_activity_count": 18,
            "account_contact_count": 3,
        })

        self.assertGreater(active["score"], inactive["score"])

    def test_multiple_known_contacts_strengthen_buying_intent(self):
        inputs = {
            "company_size": 500,
            "job_title": "Data Scientist",
            "source": "Free API",
            "inbound_text": "Evaluating a demand forecasting model.",
            "account_activity_count": 2,
        }

        single_contact = score_lead({**inputs, "account_contact_count": 1})
        multiple_contacts = score_lead({**inputs, "account_contact_count": 3})

        self.assertEqual(
            multiple_contacts["buying_intent"],
            single_contact["buying_intent"] + 8,
        )

    def test_maximum_qualification_and_priority_scores_reach_100(self):
        now = pd.Timestamp("2026-09-28T12:00:00Z")
        leads = []
        for number in range(3):
            lead = self._lead(
                f"L04{number}",
                5000,
                "UNASSIGNED",
                "Production deployment with pricing and procurement.",
            )
            lead["name"] = f"Contact {number}"
            lead["email"] = f"contact{number}@maximum.example"
            lead["company"] = "Maximum Corp"
            lead["job_title"] = "VP of Data"
            lead["source"] = "SAP"
            lead["account_activity_count"] = 5
            lead["inbound_time"] = (now - pd.Timedelta(days=7)).isoformat()
            leads.append(lead)

        view = build_lead_view(pd.DataFrame(leads), self.team, now=now)

        self.assertEqual(set(view["commercial_potential"]), {40})
        self.assertEqual(set(view["buying_intent"]), {35})
        self.assertEqual(set(view["use_case_score"]), {25})
        self.assertEqual(set(view["qualification_score"]), {100})
        self.assertEqual(set(view["qualification_points"]), {85})
        self.assertEqual(set(view["time_points"]), {15})
        self.assertEqual(set(view["score"]), {100})

    def test_maturity_and_commercial_intent_are_independent(self):
        inputs = {
            "company_size": 500,
            "job_title": "Data Scientist",
            "source": "Other",
            "account_activity_count": 1,
            "account_contact_count": 1,
        }

        baseline = score_lead({**inputs, "inbound_text": "Need more information."})
        supporting_context = score_lead({
            **inputs,
            "inbound_text": "Would like a demo and security review.",
        })
        production_context = score_lead({
            **inputs,
            "inbound_text": "Need private cloud deployment for this workflow.",
        })
        explicit_purchase = score_lead({
            **inputs,
            "inbound_text": "Please share pricing and procurement contract terms.",
        })

        self.assertEqual(
            supporting_context["buying_intent"], baseline["buying_intent"]
        )
        self.assertEqual(
            production_context["buying_intent"], baseline["buying_intent"]
        )
        self.assertEqual(production_context["use_case_score"], 25)
        self.assertEqual(
            explicit_purchase["buying_intent"], baseline["buying_intent"] + 10
        )


if __name__ == "__main__":
    unittest.main()
