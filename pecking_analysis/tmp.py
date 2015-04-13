
    ## BASE

    def summarize(self):
        '''
        Loops through all of the object's children and concatenates their
        summary data.
        :return: pandas DataFrame with concatenated summary data from all
        children
        '''
        if hasattr(self, self._children):
            value_list = list()
            for child in getattr(self, self._children):
                if hasattr(child, "_summary"):
                    columns, values = child._summary()
                    value_list.append(values)
            if len(value_list):
                return pd.DataFrame(value_list, columns=columns)


        # This should go in the bird manager and handle objs of different types,
        # skipping objects that aren't Bird, Experiment, Session, Block, or Trial.
        @staticmethod
        def summarize(objs):
            # This is hacky and will fail if all objs are not of the same type

            if len(objs):
                value_list = list()
                for obj in objs:
                    columns, values = obj._summary()
                    value_list.append(values)
                if len(value_list):
                    return pd.DataFrame(value_list, columns=columns)

            else:
                columns, values = objs._summary()
                return pd.DataFrame([values], columns=columns)


    def update(self):
        '''
        Delete
        '''
        cached = getattr(self, "_cached", [])
        for var in cached:
            if hasattr(self, var):
                delattr(self, var)

        parent = getattr(self, "_parent", None)
        if parent is not None:
            parent.update()                


                ## BIRD
        @cached_property
        def num_experiments(self):

            return len(self.experiments)

        @cached_property
        def last_experiment(self):

            if self.num_experiments > 0:
                self._sort_children()
                return self.experiments[-1]

        @cached_property
        def average_weight(self):

            return np.mean([exp.fast_weight for exp in self.experiments])

        @cached_property
        def average_peck_count(self):

            return np.mean([blk.num_trials for exp in self.experiments for
                            sess in exp.sessions for blk in sess.blocks])

        @cached_property
        def average_performance(self):

            return np.mean([blk.performance for exp in self.experiments for
                            sess in exp.sessions for blk in sess.blocks])

        @cached_property
        def significant_blocks(self):

            return 100 * np.mean([blk.is_significant for exp in
                                  self.experiments for sess in exp.sessions
                                  for blk in sess.blocks])


    ## EXPERIMENT

        @property
        def num_blocks(self):

            num_blocks = 0
            for session in self.sessions:
                num_blocks += session.num_blocks

            return num_blocks

        @property
        def total_pecks(self):

            total_pecks = 0
            for session in self.sessions:
                total_pecks += session.total_pecks

            return total_pecks

        @property
        def significant_blocks(self):

            significant_blocks = 0
            for session in self.sessions:
                significant_blocks += session.significant_blocks

            return significant_blocks

        def show_weight(self, show_plot=True):

            weights = dict()
            if self.weight is not None:
                weights.setdefault("Time", list()).append(self.fast_start)
                weights.setdefault("Weight", list()).append(self.weight)
                weights.setdefault("Label", list()).append("Fast start")
                weights.setdefault("Pecks", list()).append(None)
                weights.setdefault("Num Reward", list()).append(None)
                weights.setdefault("Food Given", list()).append(None)

            for sess in self.sessions:
                if sess.start is not None:
                    weights.setdefault("Time", list()).append(sess.start)
                    weights.setdefault("Weight", list()).append(sess.weight)
                    weights.setdefault("Label", list()).append("Session start")
                    weights.setdefault("Pecks", list()).append(None)
                    weights.setdefault("Num Reward", list()).append(None)
                    weights.setdefault("Food Given", list()).append(None)

                if sess.end is not None:
                    weights.setdefault("Time", list()).append(sess.end)
                    weights.setdefault("Weight", list()).append(sess.post_weight)
                    weights.setdefault("Label", list()).append("Session end")
                    weights.setdefault("Pecks", list()).append(sess.total_pecks)
                    weights.setdefault("Num Reward", list()).append(sess.total_reward)
                    weights.setdefault("Food Given", list()).append(sess.seed_given)

            return weights

            index = weights.pop("Time")
            df = pd.DataFrame(weights, index=index)
            df["Weight"][df["Weight"] == 0] = None
            if show_plot:
                ts = df["Weight"]
                ts.index = df.index.format(formatter=lambda x: x.strftime(
                    self.date_format))
                ax = ts.plot(rot=0)
                ax.set_xticks(range(len(ts)))
                ax.set_xticklabels(ts.index)
                ax.set_ylabel("Weight")
                ax.set_xlabel("Date")

                ax2 = ax.twinx()
                ts = df["Pecks"][~np.isnan(df["Pecks"])]
                ts.index = df.index.format(formatter=lambda x: x.strftime(
                    self.date_format))
                ts.plot(rot=0, ax=ax2)


            return df


    ## SESSION

        @property
        def total_pecks(self):

            total_pecks = 0
            for blk in self.blocks:
                total_pecks += blk.total_pecks

            return total_pecks

        @property
        def significant_blocks(self):

            significant_blocks = 0
            for blk in self.blocks:
                if blk.is_significant:
                    significant_blocks += 1

            return significant_blocks

        @property
        def total_reward(self):

            total_reward = 0
            for blk in self.blocks:
                try:
                    total_reward += blk.total_reward
                except AttributeError:
                    continue

            return total_reward

        @property
        def weight_loss(self):

            if self.experiment is not None:
                if (self.weight is not None) and (self.experiment.weight is not None):
                    return 100 * (1 - float(self.weight) / float(self.experiment.weight))


        def to_csv(self, filename):

            pd.concat([blk.data for blk in self.blocks]).to_csv(filename)


## BLOCK

    def compute_statistics(self):

        if self.data is None:
            return

        # Get peck information
        self.total_pecks = len(self.data)
        self.total_stimuli = self.total_pecks
        self.total_reward = self.data["Class"].sum()
        self.total_no_reward = self.total_pecks - self.total_reward

        # Get percentages
        self.percent_reward = self.to_percent(self.total_reward, self.total_stimuli)
        self.percent_no_reward = self.to_percent(self.total_no_reward, self.total_stimuli)

        # Add interval and interrupt data to table
        self.get_interrupts()

        grped = self.data.groupby("Class")

        # Get interruption information
        if self.total_no_reward > 0:
            self.interrupt_no_reward = grped["Interrupts"].sum()[0]
        else:
            self.interrupt_no_reward = 0
        self.percent_interrupt_no_reward = self.to_percent(self.interrupt_no_reward, self.total_no_reward)

        if self.total_reward > 0:
            self.interrupt_reward = grped["Interrupts"].sum()[1]
        else:
            self.interrupt_reward = 0
        self.percent_interrupt_reward = self.to_percent(self.interrupt_reward, self.total_reward)

        self.total_interrupt = self.interrupt_reward + self.interrupt_no_reward
        self.percent_interrupt = self.to_percent(self.total_interrupt, self.total_pecks)

        if (self.total_reward > 0) and (self.total_no_reward > 0):
            mu = (self.percent_interrupt_no_reward - self.percent_interrupt_reward)
            sigma = np.sqrt(self.percent_interrupt * (1 - self.percent_interrupt) * (1 / self.total_reward + 1 / self.total_no_reward))
            self.zscore = mu / sigma
            self.binomial_pvalue = 2 * (1 - scipy.stats.norm.cdf(np.abs(self.zscore)))
            self.is_significant = self.binomial_pvalue <= 0.05
        else:
            self.zscore = 0.0
            self.binomial_pvalue = 1.0
            self.is_significant = False

    @staticmethod
    def to_percent(value, n):

        if n != 0:
            return value / float(n)
        else:
            return 0.0

    def get_interrupts(self):

        if "Interrupts" in self.data:
            return

        time_diff = np.hstack([np.diff([ind.value for ind in self.data.index]), 0]) / 10**9
        inds = (time_diff > 0.19) & (time_diff < 6) # This value should be based on the stimulus duration

        self.data["Interrupts"] = inds
        self.data["Intervals"] = time_diff
