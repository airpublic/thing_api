from struct import unpack
from base64 import b64decode
from collections import namedtuple


LORA_FORMAT="BBHHHHHHHHHHHffBBBBBBBBBBB"


class LoraRecord(
        namedtuple(
            "LoraRecord",
            [
                "ver",
                "rtys",
                "vol",
                "afewrk1",
                "afeaux1",
                "afewrk2",
                "afeaux2",
                "afewrk3",
                "afeaux3",
                "afept1k",
                "isbwrk",
                "isbaux",
                "mics1",
                "mics2",
                "lat",
                "long",
                "rpm",
                "pm11c",
                "pm125c",
                "pm110c",
                "pm1tmp",
                "pm1hum",
                "pm21c",
                "pm225c",
                "pm210c",
                "pm2tmp",
                "pm2hum"
            ]
        )):
    pass


class RawLoraRecord(namedtuple(
        "RawLoraRecord",
        [
            "rty_ver",
            "vol",
            "afewrk1",
            "afeaux1",
            "afewrk2",
            "afeaux2",
            "afewrk3",
            "afeaux3",
            "afept1k",
            "isbwrk",
            "isbaux",
            "mics1",
            "mics2",
            "lat",
            "long",
            "rpm",
            "pm11c",
            "pm125c",
            "pm110c",
            "pm1tmp",
            "pm1hum",
            "pm21c",
            "pm225c",
            "pm210c",
            "pm2tmp",
            "pm2hum"
        ]
    )):

    @classmethod
    def unpack(klass, bytes):

        return klass._make(unpack(LORA_FORMAT, bytes))

    def decompress(self):
        return LoraRecord(
            ver=self.rty_ver << 4,
            rtys=self.rty_ver & 0b00001111,
            vol=self.vol >> 8,
            afewrk1=self.afewrk1/100.0,
            afeaux1=self.afeaux1/100.0,
            afewrk2=self.afewrk2/100.0,
            afeaux2=self.afeaux2/100.0,
            afewrk3=self.afewrk3/100.0,
            afeaux3=self.afeaux3/100.0,
            afept1k=self.afept1k/100.0,
            isbwrk=self.isbwrk/100.0,
            isbaux=self.isbaux/100.0,
            mics1=self.mics1/100.0,
            mics2=self.mics2/100.0,
            lat=self.lat,
            long=self.long,
            rpm=self.rpm*100.0,
            pm11c=self.pm11c,
            pm125c=self.pm125c,
            pm110c=self.pm110c,
            pm1tmp=self.pm1tmp*10.0,
            pm1hum=self.pm1hum*10.0,
            pm21c=self.pm21c,
            pm225c=self.pm225c,
            pm210c=self.pm210c,
            pm2tmp=self.pm2tmp*10.0,
            pm2hum=self.pm2hum*10.0
        )


def decode_payload(b64_payload):
    bin = b64decode(b64_payload)
    raw_record = RawLoraRecord.unpack(bin)
    record = raw_record.decompress()
    return record
