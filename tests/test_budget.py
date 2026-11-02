import unittest

from pramaan.budget import ModelCallBudget, ModelCallBudgetExceeded


def test_daily_budget_allows_calls_up_to_limit():
    budget = ModelCallBudget(2)
    budget.reserve()
    budget.reserve()
    with unittest.TestCase().assertRaises(ModelCallBudgetExceeded):
        budget.reserve()


def test_zero_budget_disables_hosted_calls():
    budget = ModelCallBudget(0)
    with unittest.TestCase().assertRaises(ModelCallBudgetExceeded):
        budget.reserve()
