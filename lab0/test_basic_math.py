        x1 = [2,3,5,7,8]
        x2 = [2,3,2,5,7,2,2,8]
        self.assertEqual(basic_math.skew(x1), round(scipy.stats.skew(x1), 2))
        self.assertEqual(basic_math.skew(x2), round(scipy.stats.skew(x2), 2))

        random.seed(100)
        random_floats = [random.random() for _ in range(10000)]
        x1 = [2,3,5,7,8]
        x2 = [2,3,2,5,7,2,2,8]
        self.assertEqual(basic_math.kurtosis(x1), round(scipy.stats.kurtosis(x1), 2))
        self.assertEqual(basic_math.kurtosis(x2), round(scipy.stats.kurtosis(x2), 2))

        random.seed(100)
        random_floats = [random.random() for _ in range(10000)]
