import os
import wandb
import wandb.apis.reports as wbreport

assert os.getenv('WANDB_API_KEY'), 'You must set the WANDB_API_KEY environment variable'

def model_comparison(entity: str, project: str, run_id: str):
    """Takes a run id and compares the model associated with the run to a baseline model.

    Args:
        entity (str): Name of the logged in user.
        project (str): Name of the project
        run_id (str): THe id of the run to compare

    Returns:
        str: A URL to the model comparison report
    """
    # Allow to override the args with env variables
    entity = os.getenv('WANDB_ENTITY') or entity
    project = os.getenv('WANDB_PROJECT') or project
    tag = os.getenv('BASELINE_TAG') or tag
    run_id = os.getenv('RUN_ID') or run_id

    # Path to run with model to evaluate
    candidate_run_path = os.path.join(entity, project, run_id)

    # Connect to api and get run
    api = wandb.Api()
    candidate_run = api.run(candidate_run_path)

    # Get the baseline run
    runs=api.runs(f'{entity}/{project}', {"tags": {"$in": ["baseline"]}})
    assert len(runs) == 1, 'There must be exactly one run with the tag "baseline"'
    baseline_run = runs[0]

    # New wandb session for generating report
    wandb.init(entity=entity, project=project)

    # Create report
    report = wbreport.report.Report(
        entity=entity,
        project = project,
        title='Compare runs',
        description = 'A demo of comparing runs programmatically'
    )

    # Panel grid
    pg = wbreport.PanelGrid(
        runsets=[
            wbreport.runset.Runset(entity, project, "Run Comparisons").set_filters_with_python_expr(f"ID in ['{baseline_run.id}', '{candidate_run.id}']")
        ],
        panels=[
            wbreport.RunComparer(diff_only='split', layout={'w': 24, 'h':15})
        ]
    )

    # Formatting
    report.blocks = report.blocks[:1] + [pg] + report.blocks[1:]
    report.save()

    # Write report url to github env
    if os.getenv('CI'): # is set to `true` in GitHub Actions https://docs.github.com/en/actions/learn-github-actions/variables#default-environment-variables
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f: # write the output variable REPORT_URL to the GITHUB_OUTPUT file
            print(f'REPORT_URL={report.url}', file=f)

    return report.url


if __name__ == '__main__':
    print(f'The comparison report can found at: {model_comparison()}')