import unittest
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from trae_agent.cli import cli


class TestCli(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @patch("trae_agent.cli.resolve_config_file", return_value="test_config.yaml")
    @patch("trae_agent.cli.Agent")
    @patch("trae_agent.cli.asyncio.run")
    @patch("trae_agent.cli.Config.create")
    @patch("trae_agent.cli.ConsoleFactory.create_console")
    def test_run_with_long_prompt(
        self,
        mock_create_console,
        mock_config_create,
        mock_asyncio_run,
        mock_agent_class,
        mock_resolve_config_file,
    ):
        """Test that a long prompt string is handled correctly."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.trae_agent = MagicMock()
        mock_config_create.return_value.resolve_config_values.return_value = mock_config
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        mock_console = MagicMock()
        # Add the methods that hasattr checks for
        mock_console.set_initial_task = MagicMock()
        mock_console.set_agent_context = MagicMock()
        mock_create_console.return_value = mock_console

        long_prompt = "a" * 500  # A string longer than typical filename limits
        result = self.runner.invoke(cli, ["run", long_prompt, "--working-dir", "/tmp"])
        self.assertEqual(result.exit_code, 0)

        # Verify agent.run was called with the long prompt
        mock_asyncio_run.assert_called_once()
        mock_agent.run.assert_called_once()
        args, _ = mock_agent.run.call_args
        self.assertEqual(args[0], long_prompt)

    @patch("trae_agent.cli.resolve_config_file", return_value="test_config.yaml")
    @patch("trae_agent.cli.Agent")
    @patch("trae_agent.cli.asyncio.run")
    @patch("trae_agent.cli.Config.create")
    @patch("trae_agent.cli.ConsoleFactory.create_console")
    def test_run_with_file_argument(
        self,
        mock_create_console,
        mock_config_create,
        mock_asyncio_run,
        mock_agent_class,
        mock_resolve_config_file,
    ):
        """Test that the --file argument correctly reads from a file."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.trae_agent = MagicMock()
        mock_config_create.return_value.resolve_config_values.return_value = mock_config
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        mock_console = MagicMock()
        # Add the methods that hasattr checks for
        mock_console.set_initial_task = MagicMock()
        mock_console.set_agent_context = MagicMock()
        mock_create_console.return_value = mock_console

        with self.runner.isolated_filesystem():
            with open("task.txt", "w") as f:
                f.write("task from file")

            result = self.runner.invoke(cli, ["run", "--file", "task.txt", "--working-dir", "/tmp"])
            self.assertEqual(result.exit_code, 0)

            # Verify agent.run was called with the file content
            mock_asyncio_run.assert_called_once()
            mock_agent.run.assert_called_once()
            args, _ = mock_agent.run.call_args
            self.assertEqual(args[0], "task from file")

    @patch("trae_agent.cli.resolve_config_file", return_value="test_config.yaml")
    def test_run_with_nonexistent_file(self, mock_resolve_config_file):
        """Test for a clear error when --file points to a non-existent file."""
        result = self.runner.invoke(cli, ["run", "--file", "nonexistent.txt"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Error: File not found: nonexistent.txt", result.output)

    @patch("trae_agent.cli.resolve_config_file", return_value="test_config.yaml")
    def test_run_with_both_task_and_file(self, mock_resolve_config_file):
        """Test for a clear error when both task string and --file are used."""
        result = self.runner.invoke(cli, ["run", "some task", "--file", "task.txt"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn(
            "Error: Cannot use both a task string and the --file argument.", result.output
        )

    def test_run_with_no_input(self):
        """Test for a clear error when neither task string nor --file is provided."""
        result = self.runner.invoke(cli, ["run"])
        self.assertIn("Error: Config file not found.", result.output)

    @patch("trae_agent.cli.resolve_config_file", return_value="test_config.yaml")
    @patch("trae_agent.cli.Agent")
    @patch("trae_agent.cli.Config.create")
    @patch("trae_agent.cli.ConsoleFactory.create_console")
    @patch("trae_agent.cli.os.chdir", side_effect=FileNotFoundError("No such file or directory"))
    def test_run_with_nonexistent_working_dir(
        self,
        mock_chdir,
        mock_create_console,
        mock_config_create,
        mock_agent_class,
        mock_resolve_config_file,
    ):
        """Test for a clear error when --working-dir points to a non-existent directory."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.trae_agent = MagicMock()
        mock_config_create.return_value.resolve_config_values.return_value = mock_config
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        mock_console = MagicMock()
        mock_console.set_initial_task = MagicMock()
        mock_console.set_agent_context = MagicMock()
        mock_create_console.return_value = mock_console

        result = self.runner.invoke(
            cli, ["run", "some task", "--working-dir", "/path/to/nonexistent/dir"]
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Error changing directory", result.output)

    @patch("trae_agent.cli.resolve_config_file", return_value="test_config.yaml")
    @patch("trae_agent.cli.Agent")
    @patch("trae_agent.cli.asyncio.run")
    @patch("trae_agent.cli.Config.create")
    @patch("trae_agent.cli.ConsoleFactory.create_console")
    def test_run_with_string_that_is_also_a_filename(
        self,
        mock_create_console,
        mock_config_create,
        mock_asyncio_run,
        mock_agent_class,
        mock_resolve_config_file,
    ):
        """Test that a task string that looks like a file is treated as a string."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.trae_agent = MagicMock()
        mock_config_create.return_value.resolve_config_values.return_value = mock_config
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        mock_console = MagicMock()
        # Add the methods that hasattr checks for
        mock_console.set_initial_task = MagicMock()
        mock_console.set_agent_context = MagicMock()
        mock_create_console.return_value = mock_console

        with self.runner.isolated_filesystem():
            with open("task.txt", "w") as f:
                f.write("file content")

            result = self.runner.invoke(cli, ["run", "task.txt", "--working-dir", "/tmp"])
            self.assertEqual(result.exit_code, 0)

            # Verify agent.run was called with the string "task.txt", not the file content
            mock_asyncio_run.assert_called_once()
            mock_agent.run.assert_called_once()
            args, _ = mock_agent.run.call_args
            self.assertEqual(args[0], "task.txt")

    @patch("trae_agent.cli.resolve_config_file", return_value="test_config.yaml")
    @patch("trae_agent.cli.Agent")
    @patch("trae_agent.cli.asyncio.run")
    @patch("trae_agent.cli.Config.create")
    @patch("trae_agent.cli.ConsoleFactory.create_console")
    def test_run_with_image_flag_forwards_bytes(
        self,
        mock_create_console,
        mock_config_create,
        mock_asyncio_run,
        mock_agent_class,
        mock_resolve_config_file,
    ):
        """`--image` should read the file as bytes and pass it via task_args."""
        mock_config = MagicMock()
        mock_config.trae_agent = MagicMock()
        mock_config_create.return_value.resolve_config_values.return_value = mock_config
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        mock_console = MagicMock()
        mock_console.set_initial_task = MagicMock()
        mock_console.set_agent_context = MagicMock()
        mock_create_console.return_value = mock_console

        png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"

        with self.runner.isolated_filesystem():
            with open("photo.png", "wb") as f:
                f.write(png_bytes)

            result = self.runner.invoke(
                cli,
                ["run", "describe this image", "--image", "photo.png", "--working-dir", "/tmp"],
            )
            self.assertEqual(result.exit_code, 0)

            mock_agent.run.assert_called_once()
            args, _ = mock_agent.run.call_args
            task_args = args[1]
            self.assertEqual(task_args["images"], [png_bytes])

    @patch("trae_agent.cli.resolve_config_file", return_value="test_config.yaml")
    def test_run_with_nonexistent_image_errors(self, mock_resolve_config_file):
        """`--image` pointing at a missing file should fail fast (click.Path(exists=True))."""
        result = self.runner.invoke(cli, ["run", "task", "--image", "does-not-exist.png"])
        self.assertNotEqual(result.exit_code, 0)
        # click's exists validation message includes the path
        self.assertIn("does-not-exist.png", result.output)

    @patch("trae_agent.cli.resolve_config_file", return_value="test_config.yaml")
    @patch("trae_agent.cli.Agent")
    @patch("trae_agent.cli.asyncio.run")
    @patch("trae_agent.cli.Config.create")
    @patch("trae_agent.cli.ConsoleFactory.create_console")
    def test_run_without_image_flag_passes_empty_list(
        self,
        mock_create_console,
        mock_config_create,
        mock_asyncio_run,
        mock_agent_class,
        mock_resolve_config_file,
    ):
        """Backward compat: no --image → task_args['images'] is an empty list."""
        mock_config = MagicMock()
        mock_config.trae_agent = MagicMock()
        mock_config_create.return_value.resolve_config_values.return_value = mock_config
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        mock_console = MagicMock()
        mock_console.set_initial_task = MagicMock()
        mock_console.set_agent_context = MagicMock()
        mock_create_console.return_value = mock_console

        result = self.runner.invoke(cli, ["run", "plain text task", "--working-dir", "/tmp"])
        self.assertEqual(result.exit_code, 0)

        args, _ = mock_agent.run.call_args
        self.assertEqual(args[1]["images"], [])


if __name__ == "__main__":
    unittest.main()
