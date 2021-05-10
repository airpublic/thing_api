from abc import ABC, abstractmethod


class AbstractAlertCondition(ABC):

    def __init__(self, alert, *args, **kwargs):
        self.alert = alert

    @abstractmethod
    def should_alert(self, loader, df):
        pass


class MissingData(AbstractAlertCondition):

    def should_alert(self, loader, df):

        import pdb; pdb.set_trace()


class TestCondition(AbstractAlertCondition):

    def should_alert(self, loader, df):

        return df.assign(should_alert=df['value'] > 200)
