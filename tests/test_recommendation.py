import unittest
from app.recommendation import recommend_tasks

class TestRecommendation(unittest.TestCase):
    def test_recommend_tasks(self):
        tasks = recommend_tasks("happy")
        self.assertGreater(len(tasks), 0)

if __name__ == "__main__":
    unittest.main()