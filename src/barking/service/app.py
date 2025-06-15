import asyncio

from barking.messaging.bus import EventBus

evt_name = "test"


async def consume(bus: EventBus):
    async with bus.subscribe("testing") as stream:
        async for msg in stream:
            # get the event from the msg envelope
            print(msg)


async def produce(bus: EventBus) -> None:
    coros = [
        bus.publish(
            channel="testing",
            type="test-msg",
            data={"_id": i},
        )
        for i in range(20)
    ]
    await asyncio.gather(*coros)


async def main():
    bus = EventBus()
    await bus.connect()

    tasks = [asyncio.create_task(consume(bus)), asyncio.create_task(produce(bus))]
    async with bus:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
