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
            "same_company_users": 1,
            "inbound_text": inbound_text,
            "owner_id": owner_id,
            "status": "In Progress",
        }

    def test_existing_owner_is_not_replaced_by_higher_scoring_lead(self):
        view = build_lead_view(self.leads, self.team).set_index("lead_id")

        self.assertEqual(view.at["L001", "owner_id"], "S001")
        self.assertEqual(view.at["L002", "owner_id"], "UNASSIGNED")

    def test_new_assignment_is_persisted_and_survives_reprioritization(self):
        self.leads.loc[self.leads["lead_id"] == "L001", "status"] = "Success"
        view = build_lead_view(self.leads, self.team)

        self.assertEqual(persist_owner_assignments(self.leads, view), 1)
        self.assertEqual(
            self.leads.loc[self.leads["lead_id"] == "L002", "owner_id"].iloc[0],
            "S001",
        )

        self.leads = pd.concat([
            self.leads,
            pd.DataFrame([
                self._lead("L003", 50000, "UNASSIGNED", "Production security procurement"),
            ]),
        ], ignore_index=True)
        refreshed = build_lead_view(self.leads, self.team).set_index("lead_id")

        self.assertEqual(refreshed.at["L002", "owner_id"], "S001")
        self.assertEqual(refreshed.at["L003", "owner_id"], "UNASSIGNED")

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
            "L004", 5000, "UNASSIGNED", "Production evaluation with procurement"
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
        self.assertEqual(view["waiting"], "6h 30m")
        self.assertEqual(view["time_points"], 4)

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
            "same_company_users": 1,
        })
        active = score_lead({
            "company_size": 12000,
            "job_title": "ML Platform Lead",
            "source": "Free API",
            "inbound_text": "Several teams are evaluating private networking, security, and pricing.",
            "account_activity_count": 18,
            "same_company_users": 8,
        })

        self.assertGreater(active["score"], inactive["score"])


if __name__ == "__main__":
    unittest.main()
