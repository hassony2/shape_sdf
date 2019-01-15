import datetime
from random import randint
import os
import getpass
import sys
import time
from subprocess import call

DEFAULT_QUEUES = [
    'gaia.q', 'titan.q', 'zeus.q', 'chronos.q', 'all.q', 'bigmem.q', 'gpu.q'
]
DEFAULT_TMP_DIR = '/sequoia/data1/yhasson/2019_01_13_jobs'
MAX_JOBS = 100


def generate_key():
    return '{}'.format(randint(0, 100000))


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write(
                "Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def open_parallel_task():

    # Get the tmp directory.
    username = getpass.getuser()
    tmp_dir = os.path.join(DEFAULT_TMP_DIR)

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # Generate a task id for the job.
    task_id = generate_key()

    while os.path.exists(os.path.join(tmp_dir, task_id)):
        task_id = generate_key()

    task_dir = os.path.join(tmp_dir, task_id)
    os.makedirs(task_dir)

    # Prepare script and log folder.
    sh_dir = os.path.join(task_dir, 'scripts')
    logs_dir = os.path.join(task_dir, 'logs')

    os.makedirs(sh_dir)
    os.makedirs(logs_dir)

    return task_id, task_dir


def write_submit_scripts(task, task_id, parallel_args, n_jobs, task_dir,
                         shell_var, host_name, queues, n_slots, memory,
                         memory_hard, prepend_cmd, postpend_cmd,
                         multi_threading):

    sh_dir = os.path.join(task_dir, 'scripts')
    log_dir = os.path.join(task_dir, 'logs')
    n_instances = len(parallel_args)
    instance_per_job = int((n_instances - 1) / n_jobs) + 1

    job_ids = []

    for i in range(n_jobs):
        job_ids.append(i)
        start_instance = i * instance_per_job
        end_instance = min((i + 1) * instance_per_job, n_instances)
        report_file = os.path.join(log_dir, 'report_{}.txt'.format(i))

        with open(os.path.join(sh_dir, 'submit_{}.pbs'.format(i)),
                  'w') as pbs_file:

            pbs_file.write('#$ -l mem_req={}m \n'.format(memory))
            pbs_file.write('#$ -l h_vmem={}m \n'.format(memory_hard))
            pbs_file.write('#$ -pe serial {} \n'.format(n_slots))
            if queues:
                pbs_file.write('#$ -q {}\n'.format(','.join(queues)))

            if host_name:
                pbs_file.write(
                    '#$ -l hostname={}\n'.format('|'.join(host_name)))

            pbs_file.write('#$ -e {} \n'.format(log_dir))
            pbs_file.write('#$ -o {} \n'.format(log_dir))
            pbs_file.write('#$ -N _{}_{} \n'.format(i, task_id))

            for (var, value) in shell_var:
                pbs_file.write('export {0}={1}:${0} \n'.format(var, value))

            if multi_threading:
                thread_vars = [
                    'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS'
                ]
                for thread_var in thread_vars:
                    pbs_file.write('export {0}=1 \n'.format(thread_var))

            # Write prepend commands.
            for cmd in prepend_cmd:
                pbs_file.write(cmd + '\n')

            # Write python command(s).
            for k in range(start_instance, end_instance):
                arg_string = ''
                # Add info about pyapt run to options
                parallel_args[k]['pyapt_id'] = '_{}_{}'.format(i, task_id)
                for arg in parallel_args[k]:
                    arg_val = parallel_args[k][arg]
                    if arg_val is True:
                        arg_string += '--{} '.format(arg)
                    elif isinstance(arg_val, (tuple, list)):
                        arg_string += '--{} {} '.format(
                            arg, ' '.join([str(val) for val in arg_val]))
                    else:
                        arg_string += '--{}=\'{}\' '.format(arg, arg_val)

                pbs_file.write('python {} {} >> {} \n'.format(
                    task, arg_string, report_file))

            for cmd in postpend_cmd:
                pbs_file.write(cmd + '\n')

    return job_ids


def write_metadata(task_dir, description=None):
    metadata_file = os.path.join(task_dir, 'meta.txt')
    launch_time = datetime.datetime.now()
    with open(metadata_file, 'w') as f_meta_file:
        f_meta_file.write('launch time:{}\n'.format(str(launch_time)))
        if description is not None:
            f_meta_file.write('description:'.format(description))


def generate_jobs(task_dir, job_ids):
    sh_dir = os.path.join(task_dir, 'scripts')
    job_file = os.path.join(sh_dir, 'jobs.txt')
    with open(job_file, 'w') as f_job_file:
        for job_id in job_ids:
            f_job_file.write('{}\n'.format(job_id))


def generate_launcher(task_dir, task_id, max_parallel_jobs):
    job_limit_file = os.path.join(task_dir, 'maxjob.inf')
    sh_dir = os.path.join(task_dir, 'scripts')
    job_file = os.path.join(sh_dir, 'jobs.txt')
    script = os.path.join(sh_dir, 'launcher.sh')

    # Username.
    username = getpass.getuser()

    # Setting the job limit.
    print(('Setting the max number of jobs to {} '
           '(edit {} if you wish to change that limit later)').format(
               int(max_parallel_jobs), job_limit_file))

    with open(job_limit_file, 'w') as f_job_limit:
        f_job_limit.write('{}\n'.format(int(max_parallel_jobs)))

    # Creating the job launcher.
    with open(script, 'w') as f_script:
        cmd_count = (
            '$QSTAT | grep {} | grep {} | sed "s/.* \\([0-9][0-9]*\\) *$/\\1/" '
            '| echo \\`sum=0; while read line; do let sum=$sum+$line; done; echo $sum\\` '
        ).format(username, task_id)
        f_script.write('#!/bin/sh\n')
        f_script.write('cd {}\n'.format(sh_dir))
        f_script.write('module add sge cluster-tools\n')
        f_script.write('QSTAT="qstat"\n')
        f_script.write('QSUB="qsub"\n')
        f_script.write('USER={}\n'.format(username))
        f_script.write('CODESH={}\n'.format(job_file))
        f_script.write('FOLDER={}\n'.format(sh_dir))
        f_script.write('cd $FOLDER\n')
        f_script.write('QUEUE=$1\n')
        f_script.write('SUFF=withoutSpaces.txt\n')
        f_script.write('CODESH2=$CODESH$SUFF\n')
        f_script.write('sed ' '/^$/d' ' $CODESH > $CODESH2\n')
        f_script.write('MAXJOB=$(cat {})\n'.format(job_limit_file))
        f_script.write('while read JOBID\n')
        f_script.write('do\n')
        f_script.write('    COUNTERJOBS=`{}`\n'.format(cmd_count))
        f_script.write('   if [ "$COUNTERJOBS" -ge "$MAXJOB" ]; then\n')
        f_script.write(
            ('        date "+[%d-%b-%Y %T] Task {} : '
             'Job number limit reached ($COUNTERJOBS/$MAXJOB slots used)."\n'
             ).format(task_id))
        f_script.write('       while [ "$COUNTERJOBS" -ge "$MAXJOB" ]; do\n')
        f_script.write('           sleep 10\n')
        f_script.write('           COUNTERJOBS=`{}`\n'.format(cmd_count))
        f_script.write(
            '           NEWMAXJOB=$(cat {})\n'.format(job_limit_file))
        f_script.write('           if [ "$NEWMAXJOB" -ne "$MAXJOB" ]; then\n')
        f_script.write('               MAXJOB=$NEWMAXJOB\n')
        f_script.write(
            ('               date "+[%d-%b-%Y %T] Task {} : '
             'Job number limit reached ($COUNTERJOBS/$MAXJOB slots used)."\n'
             ).format(task_id))
        f_script.write('           fi\n')
        f_script.write('        done\n')
        f_script.write('    fi\n')
        f_script.write('    date "+[%d-%b-%Y %T] Task {} : " | tr -d "\\n"\n'.
                       format(task_id))
        f_script.write(
            '    $QSUB {}\n'.format(os.path.join(sh_dir, 'submit_$JOBID.pbs')))
        f_script.write('    sleep 0.1\n')
        f_script.write('done < $CODESH2\n')

    call('chmod 711 {}'.format(script), shell=True)

    return script


def launch_jobs(task_dir,
                task_id,
                job_ids,
                max_parallel_jobs,
                ask_confirmation=True,
                only_scripts=False):

    generate_jobs(task_dir, job_ids)
    script = generate_launcher(task_dir, task_id, max_parallel_jobs)
    question = 'You are about to launch {} jobs on the cluster. Are you sure?'.format(
        len(job_ids))

    if not only_scripts and ask_confirmation and not query_yes_no(
            question, default='yes'):
        print('Cancelling the launch (files are here {})...'.format(task_dir))
        return

    if not only_scripts:
        print(
            'About to launch {} jobs on sequoia in 2s (press Ctrl+C to cancel)...'.
            format(len(job_ids)))
        time.sleep(2)
        print('Launching!')
        print(
            '========================================================================================='
        )
        _ = call('ssh sequoia {}'.format(script), shell=True)
        print(
            '========================================================================================='
        )
        print('All your jobs have now been submitted to the cluster...')
        print('You can double checks the scripts in {}/submit_*.pbs'.format(
            os.path.join(task_dir, 'scripts')))
        print('You can double checks the logs in {}/report_*.txt'.format(
            os.path.join(task_dir, 'logs')))
        print(
            'You can kill the jobs associated to that task by calling from the sequoia master node:'
        )
        print(
            'qstat -u {0} | grep {0} | grep _{1} | cut -d \' \' -f1 | xargs qdel'.
            format(getpass.getuser(), task_id))
    else:
        print('The scripts have been created here {}/submit_*.pbs'.format(
            os.path.join(task_dir, 'scripts')))
        print('To launch, ssh sequoia {}/launcher.sh'.format(
            os.path.join(task_dir, 'scripts')))


def apt_run(task,
            parallel_args,
            group_by=0,
            host_name=None,
            memory=3000,
            memory_hard=-1,
            n_jobs=0,
            n_slots=1,
            queues=DEFAULT_QUEUES,
            shell_var=None,
            prepend_cmd=None,
            postpend_cmd=None,
            max_parrallel_jobs=5,
            ask_confirmation=True,
            only_scripts=False,
            multi_threading=False,
            description=None):
    """APT run file.

    Args:
        task: name of the launch function (with path).
        parallel_args: list of dictionary containing the parameters.
        group_by: If non-zero, APT_run will approximately compute group_by sets of arguments per job. Use this parameter
            to make short jobs a bit longer so that you do not pay too much overhead for starting a job on the cluster.
        host_name: Specify nodes which should be used, e.g. use: ['node017', 'node018', 'node019', 'node020'] in
            conjonction with cluster_id set to 2 to run your jobs on the Sequoia nodes which have more memory.
            Default launch the jobs on any node.
        memory: when running jobs on the cluster you should specify the amount of memory they need in Mb.
            They will be allowed to use additional memory (up to 1.8Gb) but will be killed if they go beyond this limit.
            Please also make sure you do not request a lot more memory than you need because it will prevent other users
            to use free slots. If memory is null, it is set to the default value of 2Gb for Meleze and 3.8Gb
            for Sequoia.
        memory_hard: hard memory limit (in MB).
        n_jobs: If non-zero, APT_run will divide your function calls across n_jobs jobs on the cluster
            If null, APT_run will run one job per argument set
        n_slots: If your program uses multi-threading, use this parameter to request the proper number of slots.
        queues: list of queues that you wish to use for your jobs.
        shell_var: Use shell_var to initialize shell variables before launching your script.
            It should be a list of tuples: [('variable1', 'value1'), ('variable2', 'value2')].
        prepend_cmd: command that you want to execute before the call to your python function.
        postpend_cmd: command that you want to execute after the call to your python function.
        max_parrallel_jobs: maximum number of jobs to be launched in parallel (be careful if you use GPU).
        ask_confirmation: whether or not to ask for confirmation before launching (True by default).
        only_scripts: will only create the scripts and won't launch on the cluster (False by default).
    """

    # Deal with default parameters.
    if host_name is None:
        host_name = []

    if shell_var is None:
        shell_var = []

    if prepend_cmd is None:
        prepend_cmd = []

    if postpend_cmd is None:
        postpend_cmd = []

    # Get number of jobs for the task.
    n_instances = len(parallel_args)

    if group_by > 0 and n_jobs == 0:
        n_jobs = max(n_instances / group_by, 1)

    if n_jobs == 0:
        n_jobs = n_instances
    else:
        n_jobs = int(min(n_instances, n_jobs))

    # Create the useful folders needed for the launch.
    task_id, task_dir = open_parallel_task()

    if memory_hard == -1:
        memory_hard = memory + 1200

    # Writing the pbs scripts.
    job_ids = write_submit_scripts(task, task_id, parallel_args, n_jobs,
                                   task_dir, shell_var, host_name, queues,
                                   n_slots, memory, memory_hard, prepend_cmd,
                                   postpend_cmd, multi_threading)

    # Writing metadata
    write_metadata(task_dir)

    # Launching the jobs.
    launch_jobs(
        task_dir,
        task_id,
        job_ids,
        max_parrallel_jobs,
        ask_confirmation=ask_confirmation,
        only_scripts=only_scripts)

    return task_id
