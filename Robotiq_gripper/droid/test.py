# Standalone gripper activation test that bypasses launch_gripper.sh
# and uses a clean reset (rACT=0) -> activate (rACT=1) flow.
#
# Purpose:
# - If this script succeeds repeatedly but launch_gripper.sh fails,
#   the issue is in the launch/client code path.
# - If this script also fails, the issue is hardware / bus / state.
#
# Keep fingers and anything near the gripper clear before running.

import argparse
import sys
import time

from polymetis.robot_client.robotiq_gripper.third_party.robotiq_2finger_grippers.robotiq_2f_gripper import (
    Robotiq2FingerGripper,
)


def status_dict(gripper):
    return {
        "gACT": getattr(gripper, "gACT", None),
        "gSTA": getattr(gripper, "gSTA", None),
        "gGTO": getattr(gripper, "gGTO", None),
        "gOBJ": getattr(gripper, "gOBJ", None),
        "gFLT": getattr(gripper, "gFLT", None),
    }


def print_status(prefix, gripper, status_ok):
    s = status_dict(gripper)
    print(
        prefix,
        "status_ok=", status_ok,
        "gACT=", s["gACT"],
        "gSTA=", s["gSTA"],
        "gGTO=", s["gGTO"],
        "gOBJ=", s["gOBJ"],
        "gFLT=", s["gFLT"],
    )


def wait_for_reset(gripper, timeout_s=5.0, poll_s=0.1):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        status_ok = gripper.getStatus()
        print_status("[reset poll]", gripper, status_ok)

        if status_ok and getattr(gripper, "gACT", None) == 0:
            return True

        time.sleep(poll_s)

    return False


def wait_for_activation(gripper, timeout_s=12.0, poll_s=0.2):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        status_ok = gripper.getStatus()
        print_status("[activate poll]", gripper, status_ok)

        if status_ok and getattr(gripper, "gACT", None) == 1 and getattr(gripper, "gSTA", None) == 3:
            return True

        time.sleep(poll_s)

    return False


def run_one_cycle(gripper, cycle_idx):
    print(f"\n===== cycle {cycle_idx} =====")

    status_ok = gripper.getStatus()
    print_status("[before reset]", gripper, status_ok)

    # Step 1: clear activation bit to reset / clear faults
    gripper.deactivate_gripper()
    sent = gripper.sendCommand()
    print("[deactivate] sent =", sent)
    if not sent:
        return False, "failed to send deactivate command"

    if not wait_for_reset(gripper):
        return False, "failed to reach reset state"

    time.sleep(0.3)

    # Step 2: set activation bit to activate again
    gripper.activate_gripper()
    sent = gripper.sendCommand()
    print("[activate] sent =", sent)
    if not sent:
        return False, "failed to send activate command"

    if not wait_for_activation(gripper):
        s = status_dict(gripper)
        return (
            False,
            f"activation did not complete "
            f"(gACT={s['gACT']}, gSTA={s['gSTA']}, gGTO={s['gGTO']}, gOBJ={s['gOBJ']}, gFLT={s['gFLT']})",
        )

    print("[result] Activation completed successfully.")
    return True, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        default="/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DAT2JDK-if00-port0",
        help="Robotiq serial port",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=3,
        help="How many reset/activate cycles to run",
    )
    args = parser.parse_args()

    print("[info] opening gripper on", args.port)
    gripper = Robotiq2FingerGripper(comport=args.port)

    print("[info] init_success =", gripper.init_success)
    if not gripper.init_success:
        print("[fatal] Could not open port.")
        sys.exit(1)

    failures = 0

    for cycle_idx in range(1, args.cycles + 1):
        ok, msg = run_one_cycle(gripper, cycle_idx)
        if not ok:
            failures += 1
            print("[cycle failed]", msg)
        else:
            print("[cycle passed]")

        time.sleep(0.5)

    print("\n===== summary =====")
    print("cycles:", args.cycles)
    print("failures:", failures)

    if failures == 0:
        print("[summary] All cycles passed.")
        sys.exit(0)
    else:
        print("[summary] One or more cycles failed.")
        sys.exit(2)


if __name__ == "__main__":
    main()