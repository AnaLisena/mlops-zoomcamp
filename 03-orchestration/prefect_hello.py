from prefect import flow, task, get_run_logger


@task
def say_hello(name: str) -> str:
    message = f"Hello, {name}! Prefect is running locally."
    print(message)
    return message


@flow(name="hello-orchestration-flow")
def hello_flow(name: str = "mlops-zoomcamp") -> str:
    logger = get_run_logger()
    logger.info("Starting local Prefect hello-world flow")
    message = say_hello(name)
    logger.info("Finished local Prefect hello-world flow")
    return message


if __name__ == "__main__":
    result = hello_flow()
    print(f"Flow result: {result}")