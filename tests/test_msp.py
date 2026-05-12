import unittest
import struct
from autotune.msp.commands import MSPCommand
from autotune.msp.protocol import (
    MSPProtocol,
    MSPV1Protocol,
    MSPV2Protocol,
    create_protocol,
)


class TestMSPProtocol(unittest.TestCase):

    def test_crc8_known_values(self):
        self.assertEqual(MSPProtocol.crc8(b"\x00"), 0x00)
        self.assertEqual(MSPProtocol.crc8(b"\x01"), 0xD5)

    def test_crc8_dvb_s2(self):
        result = MSPProtocol.crc8_dvb_s2(b"\x00")
        self.assertIsInstance(result, int)
        self.assertLess(result, 256)

    def test_crc8_consistency(self):
        data = b"test_payload_1234567890"
        crc1 = MSPProtocol.crc8(data)
        crc2 = MSPProtocol.crc8(data)
        self.assertEqual(crc1, crc2)

    def test_v1_pack_msp_pid(self):
        proto = MSPV1Protocol()
        message = proto.pack(MSPCommand.MSP_PID)
        self.assertEqual(message[0], ord('$'))
        self.assertEqual(message[1], ord('M'))
        self.assertEqual(message[2], ord('<'))
        self.assertEqual(message[3], 0)
        self.assertEqual(message[4], MSPCommand.MSP_PID)
        self.assertEqual(len(message), 6)

    def test_v1_pack_with_payload(self):
        proto = MSPV1Protocol()
        payload = bytes([10, 20, 30, 40, 50])
        message = proto.pack(MSPCommand.MSP_SET_PID, payload)
        self.assertEqual(message[3], len(payload))
        self.assertEqual(message[5:5 + len(payload)], payload)
        self.assertEqual(len(message), 6 + len(payload))

    def test_v1_unpack_valid_message(self):
        proto = MSPV1Protocol()
        payload = bytes([10, 20, 30])
        message = proto.pack(MSPCommand.MSP_PID, payload)

        result = proto.unpack_header(message)
        self.assertIsNotNone(result)
        self.assertEqual(result["command"], MSPCommand.MSP_PID)
        self.assertEqual(result["payload"], payload)
        self.assertEqual(result["size"], len(payload))

    def test_v1_unpack_crc_error(self):
        proto = MSPV1Protocol()
        payload = bytes([10, 20, 30])
        message = bytearray(proto.pack(MSPCommand.MSP_PID, payload))
        message[-1] = message[-1] ^ 0xFF

        result = proto.unpack_header(bytes(message))
        self.assertIsNone(result)

    def test_v1_unpack_incomplete(self):
        proto = MSPV1Protocol()
        payload = bytes([10, 20, 30])
        message = proto.pack(MSPCommand.MSP_PID, payload)

        result = proto.unpack_header(message[:4])
        self.assertIsNone(result)

    def test_v1_unpack_empty_payload(self):
        proto = MSPV1Protocol()
        message = proto.pack(MSPCommand.MSP_STATUS)
        self.assertEqual(len(message), 6)

        result = proto.unpack_header(message)
        self.assertIsNotNone(result)
        self.assertEqual(result["payload"], b"")

    def test_v2_pack_header_format(self):
        proto = MSPV2Protocol()
        message = proto.pack(MSPCommand.MSP2_COMMON_SETTING)

        self.assertEqual(message[0], ord('$'))
        self.assertEqual(message[1], ord('X'))
        self.assertEqual(message[2], ord('<'))

    def test_v2_pack_with_payload(self):
        proto = MSPV2Protocol()
        payload = bytes([1, 2, 3, 4, 5])
        message = proto.pack(MSPCommand.MSP2_COMMON_SERIAL_CONFIG, payload)

        size = struct.unpack("<H", message[6:8])[0]
        self.assertEqual(size, len(payload))

        result = proto.unpack_header(message)
        self.assertIsNotNone(result)
        self.assertEqual(result["payload"], payload)

    def test_create_protocol_default(self):
        proto = create_protocol()
        self.assertIsInstance(proto, MSPV1Protocol)

    def test_command_enum_values(self):
        self.assertEqual(MSPCommand.MSP_PID, 112)
        self.assertEqual(MSPCommand.MSP_SET_PID, 202)
        self.assertEqual(MSPCommand.MSP_RC_TUNING, 111)
        self.assertEqual(MSPCommand.MSP_SET_RC_TUNING, 204)


class TestMSPCommandEnum(unittest.TestCase):

    def test_read_commands(self):
        read_cmds = [
            MSPCommand.MSP_STATUS,
            MSPCommand.MSP_RAW_IMU,
            MSPCommand.MSP_MOTOR,
            MSPCommand.MSP_ATTITUDE,
            MSPCommand.MSP_PID,
            MSPCommand.MSP_RC_TUNING,
        ]
        for cmd in read_cmds:
            self.assertIsInstance(cmd, int)

    def test_write_commands(self):
        write_cmds = [
            MSPCommand.MSP_SET_PID,
            MSPCommand.MSP_SET_RC_TUNING,
        ]
        for cmd in write_cmds:
            self.assertIsInstance(cmd, int)


if __name__ == "__main__":
    unittest.main()